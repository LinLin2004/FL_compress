from typing import List
import torch
from scipy.stats import norm
from .base_attack import BaseAttack, Context

class ALIE(BaseAttack):
    """
        Implements the ALIE attack.
    """

    def __init__(self, z = 0.5):
        self.z = z

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
            Generates malicious gradients to counteract honest gradients.
        """

        if self.z is None:
            num_honest = len(context.all_honest_gradients)
            num_byz = len(context.clients) - num_honest
            s = int((num_honest + num_byz) / 2 + 1) - num_byz
            self.z = norm.ppf((num_honest + num_byz - s)/(num_byz + num_honest))


        ref_grad = context.all_honest_gradients[0]
        device = ref_grad[0].device
        honest_grad = [torch.stack([client_grad[i].to(device) for client_grad in context.all_honest_gradients]) for i in range(len(ref_grad))]
        mean_grad = [torch.mean(layer_gradients, dim=0) for layer_gradients in honest_grad]
        std_grad = [torch.std(layer_gradients, dim=0) for layer_gradients in honest_grad]

        malicious_grad = [mean + self.z * std for mean, std in zip(mean_grad, std_grad)]
        return [layer_grad.reshape(ref_grad[i].shape) for i, layer_grad in enumerate(malicious_grad)]
