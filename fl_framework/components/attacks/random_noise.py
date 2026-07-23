# fl_framework/components/attacks/random_noise.py

from typing import List
import torch
from .base_attack import BaseAttack, Context

class RandomNoiseAttack(BaseAttack):
    """
    Implements a random noise attack.

    """
    def __init__(self, std: float = 30.0, seed: int = None):
        """
        Initializes the RandomNoiseAttack.

        Args:
            std (float): The standard deviation of the Gaussian noise.
            seed (int, optional): A seed for the random number generator
                                  for reproducible attacks.
        """
        self.std = std
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        self.current_step = None
        self.mean_grad = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
        Sends Gaussian noise to server.

        Args:
            gradients (List[List[torch.Tensor]]): The list of all honest gradients.

        Returns:
            List[torch.Tensor]: The list of noise.
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
    
                wList = torch.stack(
                    [g.to(device) for g in layer_gradients]
                )

                mean_tensor = torch.mean(wList, dim=0)
                self.mean_grad.append(mean_tensor.view(original_shape))
                

        corrupted_gradients = []

        for l, grad in enumerate(ref_grad):
            noise = torch.randn(
                grad.shape,
                generator=self.generator,
                device=grad.device,
                dtype=grad.dtype
            ) * self.std + self.mean_grad[l]
            corrupted_gradients.append(noise)

        return corrupted_gradients
