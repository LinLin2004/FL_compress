from typing import List
import torch

from .base_aggregator import BaseAggregator

class SumAggregator(BaseAggregator):
    """
    Modified Mean Aggregator: performs element-wise sum of gradients.

    Instead of computing the average (like in FedAvg), this aggregator
    simply sums all gradients across clients. Useful for methods that
    want to delay or scale the normalization step manually.
    """

    @torch.no_grad()
    def aggregate(
        self,
        all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Computes the element-wise sum of the given gradients.

        Args:
            all_gradients (List[List[torch.Tensor]]): A list of gradient lists,
                one for each client.

        Returns:
            List[torch.Tensor]: The summed gradient.
        """
        if not all_gradients:
            return []
        
        ref_grad = all_gradients[0]
        if not ref_grad:
            return []

        aggregated_grad = [
            torch.zeros_like(g, device=g.device) for g in ref_grad
        ]

        for client_grad in all_gradients:
            if len(client_grad) != len(aggregated_grad):
                raise ValueError("Inconsistent number of gradient tensors among clients.")
            for agg_g, client_g in zip(aggregated_grad, client_grad):
                agg_g.data.add_(client_g.to(agg_g.device))

        return aggregated_grad

    def get_state(self):
        return None

    def set_state(self, state):
        pass