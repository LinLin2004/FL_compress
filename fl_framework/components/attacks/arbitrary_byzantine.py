# fl_framework/components/attacks/arbitrary_byzantine.py
"""Arbitrary Byzantine Attack: Byzantine workers send arbitrary malicious gradients.

This attack represents the general case where Byzantine workers can send any
malicious gradient vector. The specific form of the malicious gradient is not
specified — it could be zero vectors, random noise, sign-flipped gradients,
or any other adversarial perturbation.

In this implementation, the Byzantine worker generates a random Gaussian vector
centered at the mean of honest gradients with a large standard deviation,
simulating an arbitrary malicious gradient that is not tied to any specific
attack strategy.
"""

from typing import List
import torch
from .base_attack import BaseAttack, Context


class ArbitraryByzantineAttack(BaseAttack):
    """
    Arbitrary Byzantine attack: Byzantine workers send arbitrary malicious gradients.

    This attack generates a random vector for each Byzantine worker, simulating
    the general case where the adversary can send any malicious gradient. The
    vector is drawn from a Gaussian distribution centered at the mean of honest
    gradients with a configurable standard deviation.

    Parameters
    ----------
    std : float
        The standard deviation of the Gaussian noise added to the mean honest
        gradient. Default 100.0 (large enough to be disruptive).
    seed : int, optional
        A seed for the random number generator for reproducible attacks.
    """

    # Byzantine clients do NOT need to compute honest gradients first;
    # they generate arbitrary vectors based on the honest gradients' statistics.
    if_byz_compute_grad = False

    def __init__(self, std: float = 100.0, seed: int = None):
        self.std = std
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        self.current_step = None
        self.mean_grad = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
        Generate arbitrary malicious gradients.

        Byzantine workers send arbitrary malicious gradients. In this implementation,
        we generate a Gaussian random vector centered at the mean of honest gradients
        with a large standard deviation, representing an arbitrary adversarial
        perturbation.

        Args:
            context (Context): The hook context containing all_honest_gradients.

        Returns:
            List[torch.Tensor]: The list of corrupted gradient tensors (one per layer).
        """
        ref_grad = context.all_honest_gradients[0]

        # Cache the mean gradient per step to avoid recomputation for multiple
        # Byzantine clients in the same step
        if self.current_step != context.current_step:
            self.current_step = context.current_step
            self.mean_grad = []

            num_layers = len(ref_grad)

            for i in range(num_layers):
                # Collect the i-th gradient tensor from all honest clients
                layer_gradients = [client_grad[i] for client_grad in context.all_honest_gradients]

                original_shape = layer_gradients[0].shape
                device = layer_gradients[0].device

                wList = torch.stack(
                    [g.to(device) for g in layer_gradients]
                )

                mean_tensor = torch.mean(wList, dim=0)
                self.mean_grad.append(mean_tensor.view(original_shape))

        # Generate arbitrary malicious gradient: mean + large Gaussian noise
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
