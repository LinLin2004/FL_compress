from typing import List
import torch
from .base_attack import BaseAttack, Context

class FOE(BaseAttack):
    """
    Implements the FOE attack.

    This attack manipulates a malicious client's gradient update by scaling
    the mean gradient (calculated over all honest clients) by a negative factor `mu`.
    This effectively flips the sign of the true gradient and may scale its magnitude,
    aiming to steer the global model update in the opposite and potentially disruptive direction.
    """

    def __init__(self, mu: float = -10):
        """
        Initializes the FOE attack.

        for Krum mu use e.g. -0.1 to 1 (abs small), for CwMed use e.g.-10 (abs large)

        Args:
            mu (float): The scaling factor to apply to the mean gradient.
                        Typically negative to flip the sign; magnitude controls attack strength.
        """

        self.mu = mu
        self.mean_grad = []
        self.current_step = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
        Generates the malicious gradient by flipping and scaling the sign of the mean honest gradient.

        For each training step, this method computes and caches the mean gradient from all honest clients.
        It then returns this mean gradient scaled by `mu` as the malicious update.

        Args:
            context (Context): The attack context containing the current step and the list of honest clients' gradients.

        Returns:
            List[torch.Tensor]: The list of malicious gradients for each layer, after sign flipping and scaling.
        """
        ref_grad = context.all_honest_gradients[0]
        if self.current_step != context.current_step:
            self.current_step = context.current_step
            self.mean_grad = []

            num_layers = len(ref_grad)

            # Iterate through each layer/tensor position
            for i in range(num_layers):
                # Collect the i-th gradient tensor from all clients
                layer_gradients = [client_grad[i] for client_grad in context.all_honest_gradients]

                # Store original shape and device for reconstruction
                original_shape = layer_gradients[0].shape
                device = layer_gradients[0].device

                sum_tensor = torch.zeros_like(layer_gradients[0], device=device)
                for g in layer_gradients:
                    if g.device != device:
                        sum_tensor.add_(g.to(device))
                    else:
                        sum_tensor.add_(g)

                mean_tensor = sum_tensor.div_(len(layer_gradients))
                self.mean_grad.append(mean_tensor.view(original_shape))

        return [g * self.mu for g in self.mean_grad]
