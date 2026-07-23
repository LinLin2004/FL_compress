# fl_framework/components/attacks/sign_flipping.py

from typing import List
import torch
from .base_attack import BaseAttack, Context

class ZeroGradientAttack(BaseAttack):
    """
        Implements the Zero Gradient attack.

        This attack aims to disrupt training by generating malicious gradients
        that cancel out the expected update. It computes a scaled negative sum
        of honest gradients divided by the number of Byzantine clients, then
        returns the negation of this average as attack gradients.

        Note: The attack relies on the number of Byzantine clients to scale the sum.
    """

    def __init__(self):
        self.mean_grad = []
        self.current_step = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
            Generates malicious gradients to counteract honest gradients.

            On a new training step, computes a scaled sum of honest client gradients
            divided by the number of Byzantine clients (assumed to be Byzantine count),
            then returns their negation to effectively diminish or zero out the aggregate.

            Args:
                context (Context): The attack context containing all gradients and honest gradients.

            Returns:
                List[torch.Tensor]: The forged malicious gradients to be sent by Byzantine clients.
        """
        ref_grad = context.all_honest_gradients[0]
        num_byzantine = len(context.grad) - len(context.all_honest_gradients)
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
    
                wList = torch.stack(
                    [g.to(device) for g in layer_gradients]
                )

                mean_tensor = torch.sum(wList, dim=0) / num_byzantine
                self.mean_grad.append(mean_tensor.view(original_shape))
        
        return [g * -1 for g in self.mean_grad]
