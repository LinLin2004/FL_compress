import torch
from typing import List, Dict, Any
from .base_aggregator import BaseAggregator

class KrumAggregator(BaseAggregator):
    """
    Implements the Krum aggregation strategy.
 
    Krum is a Byzantine-robust aggregation rule that selects one client's
    gradient that is "closest" to its neighbors. It is designed to defend
    against a certain number of malicious clients (`byzantine_size`).
 
    The score for each client is the sum of its squared Euclidean distances
    to its `n - f - 1` nearest neighbors, where `n` is the total number of
    clients and `f` is the number of Byzantine clients. The client with the
    lowest score is selected.
 
    """
 
    def __init__(self, num_byzantine: int):
        """
        Initializes the KrumAggregator.
 
        Args:
            byzantine_size (int): The number of Byzantine (malicious) clients
                the algorithm is designed to tolerate. Often denoted as `f`.
                A crucial prerequisite for Krum is that the total number of
                clients `n` must be greater than `2 * f`.
        """
        super().__init__()
        if num_byzantine < 0:
            raise ValueError("byzantine_size cannot be negative.")
        self.byzantine_size = num_byzantine
 
    @torch.no_grad()
    def aggregate(
        self,
        all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Selects one client's gradient using the Krum algorithm.
 
        Args:
            all_gradients (List[List[torch.Tensor]]): A list of gradient lists,
                one for each client.
 
        Returns:
            List[torch.Tensor]: The selected client's gradient list.
        """
        # --- Input Validation ---
        if not all_gradients:
            return []
 
        num_clients = len(all_gradients)
        if num_clients == 0:
            return []
 
        # Krum's theoretical guarantee requires n > 2f
        if num_clients <= 2 * self.byzantine_size:
            raise ValueError(
                f"Number of clients ({num_clients}) must be greater than "
                f"2 * byzantine_size ({2 * self.byzantine_size})."
            )
 
        # --- Aggregation Logic ---
        # 1. Flatten each client's entire list of gradients into a single vector
        device = all_gradients[0][0].device
        flat_gradients = torch.stack([
            torch.cat([g.view(-1) for g in client_grad]).to(device)
            for client_grad in all_gradients
        ])
 
        # 2. Calculate pairwise squared Euclidean distances
        # This is a much more efficient, vectorized way than nested for-loops.
        distances = torch.cdist(flat_gradients, flat_gradients, p=2) ** 2
 
        # 3. For each client, find the distances to its k-1 closest neighbors
        # The number of neighbors to consider is n - f - 1.
        # This includes the distance to itself (which is 0).
        k = num_clients - self.byzantine_size - 1
        
        # Take the top k smallest
        top_k_dists, _ = torch.topk(distances, k, dim=1, largest=False)
 
        # 4. Calculate the score for each client (sum of these distances)
        scores = torch.sum(top_k_dists, dim=1)
 
        # 5. Find the index of the client with the minimum score
        best_client_idx = torch.argmin(scores)
 
        # 6. Return the gradients of the selected client
        aggregated_grad = [g.to(device) for g in all_gradients[best_client_idx]]
        # print(best_client_idx)
        return aggregated_grad
 
    def get_state(self) -> Dict[str, Any]:
        return None
 
    def set_state(self, state: Dict[str, Any]):
        pass