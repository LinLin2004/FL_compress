from typing import List
import torch
from .base_aggregator import BaseAggregator

class MedianAggregator(BaseAggregator):
    """
    Implements the element-wise Median aggregation strategy.

    This aggregator computes the median for each element of the gradient tensors
    across all clients. It is robust against a certain fraction of Byzantine
    clients who might send outlier gradients.
    """

    @torch.no_grad()
    def aggregate(self, all_gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Aggregate gradients by computing the element-wise median.

        Args:
            all_gradients (List[List[torch.Tensor]]): A list where each element
                is a list of gradient tensors from a single client.

        Returns:
            List[torch.Tensor]: The aggregated gradient, where each tensor is the
                element-wise median of the corresponding tensors from all clients.
        """
        if not all_gradients:
            return []
        
        num_clients = len(all_gradients)
        if num_clients == 0:
            return []
 
        # Use the first client's gradients as a reference for shape and device
        ref_grad = all_gradients[0]

        if not ref_grad:
            return []
        
        num_layers = len(ref_grad)
        median_gradient = []

        # Iterate through each layer/tensor position
        for i in range(num_layers):
            # Collect the i-th gradient tensor from all clients
            layer_gradients = [client_grad[i] for client_grad in all_gradients]
 
            # Store original shape and device for reconstruction
            original_shape = layer_gradients[0].shape
            device = layer_gradients[0].device
            # Flatten each tensor and stack them. Each row in `wList`
            # represents a client's flattened gradient for the current layer.
            wList = torch.stack(
                [g.to(device).view(-1) for g in layer_gradients]
            )

            # Compute the median along the client dimension (dim=0)
            median_tensor = torch.median(wList, dim=0).values
            median_gradient.append(median_tensor.view(original_shape))

        return median_gradient
    
    def get_state(self):
        return None
    
    def set_state(self):
        pass