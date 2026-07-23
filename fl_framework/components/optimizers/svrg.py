import torch
from typing import Dict
from fl_framework.components.optimizers.base_optimizer import BaseOptimizer
from fl_framework.core.hooks import hook_registry, Context, HookType
from copy import deepcopy

class SVRG(BaseOptimizer):
    """
    Katyusha optimizer adapted for the hook-based framework.

    This optimizer implements the Katyusha algorithm, a variance-reduced method
    that uses a two-loop structure to achieve faster convergence.
    """

    def __init__(self, lr: float, weight_decay: float = 0.0001):
        """
        Initializes the Katyusha optimizer.

        Args:
            lr (float): The learning rate (gamma in the original paper).
            round_steps (int): The number of steps in a round
            tau1 (float): The first momentum parameter for Katyusha.
            tau2 (float): The second momentum parameter for Katyusha.
        """
        super().__init__(lr)
        self.weight_decay = weight_decay

        self.clients_full_grad = None
        self.tilde_x_models = dict()

    def register_hooks(self):
        """
        Registers all hooks required for the Katyusha algorithm's lifecycle.
        """
        hook_registry.register(HookType.BEFORE_RUN, self._initialize_state)
        hook_registry.register(HookType.BEFORE_ROUND_CLIENT, self._compute_full_gradient)
        hook_registry.register(HookType.AFTER_COMPUTE, self._compute_corrected_grad)

    def _initialize_state(self, context: Context):
        """
        Hook for BEFORE_TRAINING_BEGINS.
        Initializes all state tensors based on the model's initial parameters.
        """
        self.tilde_x_models = deepcopy(context.server.models_on_devices)
        self.clients_full_grad = [[] for _ in context.clients]

        

    def _compute_full_gradient(self, context: Context):
        """
        Hook for BEFORE_ROUND_CLIENT.
        Computes the full gradient.
        """
        client = context.clients[context.current_client_id]
        client_id = context.current_client_id
        self.clients_full_grad[client_id] = []
        if client.client_type == 'Byzantine':
            return

        device = client.device
        model = self.tilde_x_models[device]
        loader = client.sampler.dataloader if hasattr(client.sampler, 'dataloader') else client.sampler.dataset
        num_batch = len(loader)


        model.train()
        model.zero_grad()
        for _, data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = client.loss_fn(output, target)
            loss.backward()
        
        for par in model.parameters():
            self.clients_full_grad[client_id].append(
                par.grad.detach().clone() / num_batch
            )
        model.zero_grad()


    def _compute_corrected_grad(self, context: Context):
        """
        Hook for AFTER_COMPUTE.
        Corrects the raw gradient using the variance reduction formula.
        """
        client_id = context.current_client_id
        client = context.clients[client_id]
        if client.client_type == 'Byzantine':
            return


        # Compute gradient using tilde x
        device = client.device
        model = client.model
        tilde_model = self.tilde_x_models[device]
        data, target = context.data, context.target
        tilde_model.zero_grad()
        tild_output = tilde_model(data)
        tild_loss = client.loss_fn(tild_output, target)
        tild_loss.backward()

        # Apply weight decay to the new gradient
        if self.weight_decay > 0:
            for i, p in enumerate(model.parameters()):
                p.grad.data.add_(p.data, alpha=self.weight_decay)
                context.grad[client_id][i].data.copy_(p.grad)

        # Compute corrected gradient
        for x, tilde_x, mu, grad_to_send in zip(client.model.parameters(), tilde_model.parameters(), self.clients_full_grad[client_id], context.grad[client_id]):
            corrected_grad = mu + x.grad - tilde_x.grad
            x.grad.data.copy_(corrected_grad)
            grad_to_send.data.copy_(corrected_grad)

    @torch.no_grad()
    def step(self, server,  aggregated_grad: Context):
        """
        Performs updates.
        """
        model = server.model

        # Ensure gradients are on the same device as the model parameters
        # and perform the SGD-like update step manually.
        for param, aggregated_grad in zip(model.parameters(), aggregated_grad):
            # Move grad to the correct device just in case
            grad_on_device = aggregated_grad.to(param.device)
            param.data.sub_(grad_on_device * self.lr)
        
    def get_state(self) -> Dict:
        """
        Returns the current state of the optimizer for checkpointing.
        """
        return {
            'lr': self.lr,
            'tilde_x': next(iter(self.tilde_x_models.values())).state_dict(),
            'full_grad': self.clients_full_grad,
        }

    def set_state(self, state: Dict):
        """
        Sets the state of the optimizer from a checkpoint.
        """
        # Load hyperparameters
        self.lr = state['lr']
        self.clients_full_grad = state['full_grad']

        for model in self.tilde_x_models.values():
            model.load_state_dict(state['tilde_x'])
        