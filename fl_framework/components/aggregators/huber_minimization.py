import torch
from typing import List
from .base_aggregator import BaseAggregator
from copy import deepcopy

class HuberAggregator(BaseAggregator):
    """
    Implements a robust gradient aggregation strategy based on Huber loss minimization.

    This aggregator treats gradients from all clients as points in high-dimensional space
    for each layer independently, and computes an aggregated gradient by minimizing
    the sum of Huber losses. The Huber loss combines the squared loss (for small residuals)
    and absolute loss (for large residuals), which makes the aggregation robust to outliers,
    such as malicious or Byzantine clients.

    The minimization is performed iteratively using an iterative re-weighted update scheme.
    """
 
    def __init__(self, max_iter: int = 80, tol: float = 1e-5, thr: float = 0.2):
        """
        Initializes the HuberAggregator.

        Args:
            max_iter (int): Maximum iterations for the iterative minimization.
            tol (float): Tolerance for convergence; stops if update magnitude is below this.
            thr (float): Huber loss threshold defining the boundary between quadratic and linear loss.
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.thr = thr
 
    @torch.no_grad()
    def aggregate(
        self,
        all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Computes the robust element-wise aggregated gradient using Huber loss minimization.

        Args:
            all_gradients (List[List[torch.Tensor]]): A list where each element is 
                the list of gradient tensors from one client.

        Returns:
            List[torch.Tensor]: Aggregated gradients, one tensor per layer.
        """
        # --- Input Validation ---
        if not all_gradients:
            return []
        
        num_clients = len(all_gradients)
        if num_clients == 0:
            return []
 
        # Use the first client's gradients as a reference for shape and device
        ref_grad = deepcopy(all_gradients[0])
        if not ref_grad:
            return []
        
        device = ref_grad[0].device
        aggregated_grad = []
 
        # --- Aggregation Logic ---
        # Iterate over each layer/tensor in the gradient list
        flat_gradients = torch.stack([
            torch.cat([g.view(-1) for g in client_grad]).to(device)
            for client_grad in all_gradients
        ])
        huber_min_grad = self._compute_huber_minimization(flat_gradients)
        aggregated_grad = []

        cum = 0
        for g in ref_grad:
            aggregated_grad.append(huber_min_grad[cum:cum+g.view(-1).shape[0]].view(g.shape))
            cum += g.view(-1).shape[0]

        return aggregated_grad
    
    def _huber_loss_fun(self, x: torch.Tensor, thr: float):
        """
        huber loss function used in our aggregation
        """
        out = torch.where(
            x < thr,
            x * x / 2,
            thr * x - thr**2 / 2
        )
        return out
 
    def _compute_huber_minimization(self, wList: torch.Tensor) -> torch.Tensor:
        """
        Computes the huber loss minimization for a set of vectors.
 
        Args:
            wList (torch.Tensor): A 2D tensor where each row is a vector
                (a client's flattened gradient for one layer).
 
        Returns:
            torch.Tensor: The aggregated vector.
        """
        # Initialize the guess with the element-wise mean
        guess = torch.mean(wList, dim=0)
 
        # Iteratively refine the guess
        for i in range(self.max_iter):
            # Calculate Euclidean distances from all points to the current guess
            distances = torch.norm(guess - wList, dim=1)
 
            # Avoid division by zero: if a point is the same as the guess,
            # its distance is 0. We replace 0 with a small number.
            distances[distances == 0] = 1e-10
 
            weight = self.thr / distances
            weight[weight > 1] = 1

            guess_next = torch.sum(weight.unsqueeze(1) * wList, dim=0) / torch.sum(weight, dim=0)

            # Check for convergence
            guess_movement = torch.norm(guess - guess_next)
            guess = guess_next
            if guess_movement <= self.tol:
                break
        # print(i)
        return guess

    def get_state(self):
        return None
    
    def set_state(self, state):
        pass