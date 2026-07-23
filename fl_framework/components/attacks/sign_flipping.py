from typing import List
import torch
from .base_attack import BaseAttack, Context

class SignFlipping(BaseAttack):
    """
    Implements the Sign Flipping attack.

    This attack flips the sign of the Byzantine client's own computed gradient,
    effectively steering the global model update in the opposite direction.
    """

    if_byz_compute_grad = True

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
        Generates the malicious gradient by flipping the sign of the Byzantine client's own gradient.

        Args:
            context (Context): The attack context containing the current client's gradient.

        Returns:
            List[torch.Tensor]: The list of malicious gradients for each layer, after sign flipping.
        """
        current_id = context.current_client_id
        return [-1 * g for g in context.grad[current_id]]
