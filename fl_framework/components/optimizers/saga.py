import torch
from tqdm import tqdm
from typing import Dict

from fl_framework.components.optimizers.base_optimizer import BaseOptimizer
from fl_framework.core.hooks import Context, HookType, hook_registry

class SAGA(BaseOptimizer):
    """
    SAGA optimizer adapted for the hook-based federated learning framework.

    This optimizer implements the SAGA algorithm, a variance-reduced method that
    maintains a table of historical gradients for each data point to accelerate
    convergence. The state (gradient table and its average) is stored distributively
    on each client.
    """

    def __init__(self, lr: float, weight_decay: float = 0.001):
        """
        Initializes the SAGA optimizer.

        Args:
            lr (float): The learning rate.
            weight_decay (float): L2 regularization factor.
        """
        super().__init__(lr)
        self.weight_decay = weight_decay
        self.grad_table = {}
        self.clients_avg_grad = []

    def register_hooks(self):
        """
        Registers all hooks required for the SAGA algorithm's lifecycle.
        """
        # This hook initializes the gradient table on each client at the very beginning.
        hook_registry.register(HookType.BEFORE_RUN, self._initialize_client_state)
        # This is the core hook for SAGA's gradient correction and state update.
        hook_registry.register(HookType.AFTER_COMPUTE, self._compute_corrected_grad_and_update_state)

    def _initialize_client_state(self, context: Context):
        """
        Hook for BEFORE_RUN. Initializes the gradient store and average gradient on all clients.
        This is a one-time, potentially expensive operation.
        The framework must trigger this hook for each client before training starts.
        """

        self.grad_table = {}
        self.clients_avg_grad = []
        for client in tqdm(context.clients, desc="Initializing SAGA gradient table"):
            if client.client_type == 'Byzantine':
                continue

            # The client needs its own model copy to compute gradients without interference
            model = client.model
            device = client.device
            model.train()

            # Use a dataloader to iterate through the client's entire dataset
            loader = client.sampler.dataloader if hasattr(client.sampler, 'dataloader') else client.sampler.dataset
            _, data, _ = next(iter(loader))
            assert data.shape[0] == 1, 'Only batchsize == 1 is supported in SAGA'

            client_grad_avg = [torch.zeros_like(p).cpu() for p in model.parameters()]
            num_samples = len(loader)
            for i, data, target in loader:
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                output = model(data)
                loss = client.loss_fn(output, target)
                loss.backward()
                
                key = str(int(i.item())) + str(client)
                self.grad_table[key] = []
                for p, layer_grad_avg in zip(model.parameters(), client_grad_avg):
                    self.grad_table[key].append(p.grad.clone().cpu())
                    layer_grad_avg.data.add_(p.grad.cpu() / num_samples)
            self.clients_avg_grad.append(client_grad_avg)
            model.zero_grad()

    @torch.no_grad()
    def _compute_corrected_grad_and_update_state(self, context: Context):
        """
        Hook for AFTER_COMPUTE. Computes SAGA's corrected gradient and updates the client's state.
        """
        client_id = context.current_client_id
        client = context.clients[client_id]
        if client.client_type == 'Byzantine':
            return

        model = client.model

        # Get the newly computed gradient for the current sample
        new_grad = context.grad[client_id]
        
        # Apply weight decay to the new gradient
        if self.weight_decay > 0:
            for i, p in enumerate(model.parameters()):
                new_grad[i].data.add_(p.data, alpha=self.weight_decay)
                p.grad.data.copy_(new_grad[i])

        # Compute the corrected SAGA gradient
        # g = new_grad - old_grad + avg_grad
        # update old_grad
        key = str(int(context.index.item())) + str(client)
        old_grad = self.grad_table[key]
        print(old_grad.sum())
        avg_grad = self.clients_avg_grad[client_id]
        for n_g, o_g, a_g, send_g in zip(new_grad, old_grad, avg_grad, context.grad[client_id]):
            send_g.data.copy_(n_g - o_g.to(n_g.device) + a_g.to(n_g.device))
        

        # Update the client's state
        # Update the average gradient: avg_new = avg_old + (new_grad - old_grad) / N
        num_samples = len(client.sampler)
        for i in range(len(avg_grad)):
            avg_grad[i].data.add_(new_grad[i].cpu() - old_grad[i].cpu(), alpha=1.0 / num_samples)
        
        # Update the gradient store with the new gradient
        for n_g, o_g in zip(new_grad, old_grad):
            o_g.data.copy_(n_g.cpu())
        print(old_grad[0].sum())
    
    @torch.no_grad()
    def step(self, server, aggregated_grad) -> None:
        """
        Performs the global model update based on aggregated gradients.
        """
        model = server.model

        # Ensure gradients are on the same device as the model parameters
        # and perform the SGD-like update step manually.
        for param, aggregated_grad in zip(model.parameters(), aggregated_grad):
            # Move grad to the correct device just in case
            grad_on_device = aggregated_grad.to(param.device)
            param.data.sub_(grad_on_device * self.lr)

    def get_state(self) -> Dict:
        # state = dict(
        #     grad_table=self.grad_table,
        #     clients_avg_grad=self.clients_avg_grad
        # )
        # return state
        return

    def set_state(self, state: Dict):
        # self.grad_table = state['grad_table']
        # self.clients_avg_grad = state['clients_avg_grad']
        pass
