# fl_framework/components/aggregators/mean.py

from typing import List
import torch

from .base_aggregator import BaseAggregator

class MeanAggregator(BaseAggregator):
    """
    Implements the standard mean aggregation strategy.

    This aggregator calculates the element-wise average of the gradients
    provided by a list of clients. It is the default aggregation method
    used in the classic FedAvg algorithm.
    """

    @torch.no_grad()
    def aggregate(
        self,
        all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Computes the element-wise mean of the given gradients.

        Args:
            all_gradients (List[List[torch.Tensor]]): A list of gradient lists,
                one for each client.

        Returns:
            List[torch.Tensor]: The averaged gradient.
        """
        # --- Input Validation ---
        if not all_gradients:
            return []
        
        num_clients = len(all_gradients)
        if num_clients == 0:
            return []

        # Use the first client's gradients as a reference for shape and device
        ref_grad = all_gradients[0]
        if not ref_grad:
            return []

        # --- Aggregation Logic ---
        # 1. Initialize a list of zero tensors with the correct shape and device
        aggregated_grad = [
            torch.zeros_like(g, device=g.device) for g in ref_grad
        ]

        # 2. Sum up all gradients from all clients
        for client_grad in all_gradients:
            # Ensure each client provided a consistent number of gradient tensors
            if len(client_grad) != len(aggregated_grad):
                raise ValueError("Inconsistent number of gradient tensors among clients.")

            for agg_g, client_g in zip(aggregated_grad, client_grad):
                # Ensure gradients are on the same device before adding
                agg_g.data.add_(client_g.to(agg_g.device))
        
        for agg_g in aggregated_grad:
            agg_g.data.div_(num_clients)

        return aggregated_grad

    def get_state(self):
        return None
    
    def set_state(self, state):
        pass