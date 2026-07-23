import torch
from abc import ABC, abstractmethod
from typing import Dict
from fl_framework.components.optimizers.base_optimizer import BaseOptimizer
from fl_framework.core.hooks import hook_registry, Context, HookType
from copy import deepcopy

class Katyusha(BaseOptimizer):
    """
    Katyusha optimizer adapted for the hook-based framework.

    This optimizer implements the Katyusha algorithm, a variance-reduced method
    that uses a two-loop structure to achieve faster convergence.
    """

    def __init__(self, lr: float, round_steps: int, tau1: float = 0.1, tau2: float = 0.2, weight_decay: float = 0.000):
        """
        Initializes the Katyusha optimizer.

        Args:
            lr (float): The learning rate (gamma in the original paper).
            round_steps (int): The number of steps in a round
            tau1 (float): The first momentum parameter for Katyusha.
            tau2 (float): The second momentum parameter for Katyusha.
        """
        super().__init__(lr)
        # if not 0 < tau1:
        #     raise ValueError(f"Invalid tau1 value: {tau1}, must be > 0")
        # if not 0 < tau2:
        #     raise ValueError(f"Invalid tau2 value: {tau2}, must be > 0")

        self.tau1 = tau1
        self.tau2 = tau2
        self.weight_decay = weight_decay

        # --- State variables, initialized in the `_initialize_state` hook ---
        self.z = []
        self.y = []
        self.tilde_x_models = dict()
        self.next_tilde_x = []
        self.clients_full_grad = []
        self.round_steps = round_steps

    def register_hooks(self):
        """
        Registers all hooks required for the Katyusha algorithm's lifecycle.
        """
        hook_registry.register(HookType.BEFORE_RUN, self._initialize_state)
        hook_registry.register(HookType.BEFORE_ROUND_CLIENT, self._compute_full_gradient)
        hook_registry.register(HookType.BEFORE_STEP_SERVER, self._prepare_step_params)
        hook_registry.register(HookType.AFTER_COMPUTE, self._compute_corrected_grad)
        hook_registry.register(HookType.AFTER_ROUND_SERVER, self._update_tilde_model)

    @torch.no_grad()
    def _initialize_state(self, context: Context):
        """
        Hook for BEFORE_RUN.
        Initializes all state tensors based on the model's initial parameters.
        """
        # Get initial parameters from the model provided by the framework context
        
        self.tilde_x_models = deepcopy(context.server.models_on_devices)

        server_model = context.server.model
        for x in server_model.parameters():
            self.y.append(x)
            self.z.append(x)
            self.next_tilde_x.append(torch.zeros_like(x))
        
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


    @torch.no_grad()
    def _prepare_step_params(self, context: Context):
        """
        Hook for BEFORE_STEP_SERVER.
        Calculates the `x_k` for the current step.
        """
        server = context.server
        model = server.model
        device = server.device

        for x, tilde_x, z, y in zip(model.parameters(), self.tilde_x_models[device].parameters(), self.z, self.y):
            x_next = self.tau1 * z + self.tau2 * tilde_x + (1 - self.tau1 - self.tau2) * y
            x.data.copy_(x_next)

        # distribute x_k to all clients
        server.distribute_model()


    def _compute_corrected_grad(self, context: Context):
        """
        Hook for AFTER_COMPUTE.
        Corrects the raw gradient using the variance reduction formula.
        """
        client_id = context.current_client_id
        client = context.clients[client_id]
        if client.client_type == 'Byzantine':
            return

        # Apply weight decay to the new gradient
        model = client.model
        if self.weight_decay > 0:
            for i, p in enumerate(model.parameters()):
                p.grad.data.add_(p.data, alpha=self.weight_decay)
                context.grad[client_id][i].data.copy_(p.grad)

        # Compute gradient using tilde x
        device = client.device
        tilde_model = self.tilde_x_models[device]
        data, target = context.data, context.target
        tilde_model.zero_grad()
        tild_output = tilde_model(data)
        tild_loss = client.loss_fn(tild_output, target)
        tild_loss.backward()

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
        agg_grad = aggregated_grad
        server = server

        for z, y, g, x, next_tilde_x in zip(self.z, self.y, agg_grad, server.model.parameters(), self.next_tilde_x):
            z.data.sub_(self.lr * g)
            y.data.copy_(x - self.tau1 * self.lr * g)
            next_tilde_x.data.add_(y / self.round_steps)
        
    @torch.no_grad()
    def _update_tilde_model(self, context: Context):
        """
        Hook for AFTER_ROUND.
        Updates the anchor point `tilde_x`.
        """
        tilde_x_model = next(iter(self.tilde_x_models.values()))
        for tilde_x, next_tilde_x in zip(tilde_x_model.parameters(), self.next_tilde_x):
            tilde_x.data.copy_(next_tilde_x)
            next_tilde_x.data.zero_()

        state_dict = tilde_x_model.state_dict()
        for m in self.tilde_x_models.values():
            m.load_state_dict(state_dict)
    
    @torch.no_grad()
    def _out_model(self, context: Context):
        """
        Hook for AFTER_RUN.
        Compute x_out
        """
        server = context.server
        model = server.model
        device = server.device

        for x, tilde_x, y in zip(model.parameters(), self.tilde_x_models[device].parameters(), self.y):
            x_out = self.tau2 * self.round_steps * tilde_x + (1 - self.tau1 - self.tau2) * y / \
                self.tau2 * self.round_steps + (1 - self.tau1 - self.tau2)
            x.data.copy_(x_out)
        
        server.distribute_model()


    def get_state(self) -> Dict:
        """
        Returns the current state of the optimizer for checkpointing.
        """
        return {
            'lr': self.lr,
            'tau1': self.tau1,
            'tau2': self.tau2,
            'z': self.z,
            'y': self.y,
            'tilde_x': next(iter(self.tilde_x_models.values())).state_dict(),
            'next_tilde_x': self.next_tilde_x,
            'full_grad': self.clients_full_grad,
        }

    def set_state(self, state: Dict):
        """
        Sets the state of the optimizer from a checkpoint.
        """
        # Load hyperparameters
        self.lr = state['lr']
        self.tau1 = state['tau1']
        self.tau2 = state['tau2']
        
        self.z = state['z']
        self.y = state['y']
        self.next_tilde_x = state['next_tilde_x']
        self.clients_full_grad = state['full_grad']

        for model in self.tilde_x_models.values():
            model.load_state_dict(state['tilde_x'])
        