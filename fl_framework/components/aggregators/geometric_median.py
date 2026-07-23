import torch
from typing import List
from .base_aggregator import BaseAggregator

class GeometricMedianAggregator(BaseAggregator):
    """
    Implements the geometric median aggregation strategy.
 
    This aggregator computes the geometric median for each gradient tensor across
    all clients. The geometric median is a robust estimator of the central
    tendency, making it less sensitive to outliers (i.e., malicious or
    Byzantine clients) compared to the standard mean.
 
    The calculation is performed iteratively using Weiszfeld's algorithm for
    each layer's gradients independently.
    """
 
    def __init__(self, max_iter: int = 80, tol: float = 1e-5):
        """
        Initializes the GeometricMedianAggregator.
 
        Args:
            max_iter (int): The maximum number of iterations for the Weiszfeld's
                algorithm.
            tol (float): The tolerance for convergence. The algorithm stops
                if the L2 norm of the update is less than this value.
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
 
    @torch.no_grad()
    def aggregate(
        self,
        all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Computes the element-wise geometric median of the given gradients.
 
        This is done by treating each layer's gradients from all clients as a
        set of points in a high-dimensional space and finding their
        geometric median.
 
        Args:
            all_gradients (List[List[torch.Tensor]]): A list of gradient lists,
                one for each client.
 
        Returns:
            List[torch.Tensor]: The aggregated gradient based on geometric median.
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
        
        num_layers = len(ref_grad)
        aggregated_grad = []
 
        # --- Aggregation Logic ---
        # Iterate over each layer/tensor in the gradient list
        for i in range(num_layers):
            # 1. Collect the i-th gradient tensor from all clients
            layer_gradients = [client_grad[i] for client_grad in all_gradients]
 
            # Store original shape and device for reconstruction
            original_shape = layer_gradients[0].shape
            device = layer_gradients[0].device
 
            # 2. Flatten each tensor and stack them. Each row in `wList`
            # represents a client's flattened gradient for the current layer.
            wList = torch.stack(
                [g.to(device).view(-1) for g in layer_gradients]
            )
 
            # 3. Compute the geometric median for the current layer's gradients
            layer_median = self._compute_geometric_median(wList)
 
            # 4. Reshape the result back to its original shape and append
            aggregated_grad.append(layer_median.view(original_shape))
 
        return aggregated_grad
 
    def _compute_geometric_median(self, wList: torch.Tensor) -> torch.Tensor:
        """
        Computes the geometric median for a set of vectors.
 
        Args:
            wList (torch.Tensor): A 2D tensor where each row is a vector
                (a client's flattened gradient for one layer).
 
        Returns:
            torch.Tensor: The geometric median vector.
        """
        # Initialize the guess with the element-wise mean
        guess = torch.mean(wList, dim=0)
 
        # Iteratively refine the guess using Weiszfeld's algorithm
        for _ in range(self.max_iter):
            # Calculate Euclidean distances from all points to the current guess
            distances = torch.norm(wList - guess, dim=1)
 
            # Avoid division by zero: if a point is the same as the guess,
            # its distance is 0. We replace 0 with a small number.
            distances[distances == 0] = 1e-10
 
            # Calculate the weights (inverse of distances)
            weights = 1.0 / distances
 
            # Update the guess: a weighted average of the points
            numerator = torch.sum(wList * weights.unsqueeze(1), dim=0)
            denominator = torch.sum(weights)
            
            guess_next = numerator / denominator
 
            # Check for convergence
            guess_movement = torch.norm(guess - guess_next)
            guess = guess_next
            if guess_movement <= self.tol:
                break
        
        return guess

    def get_state(self):
        return None
    
    def set_state(self, state):
        pass