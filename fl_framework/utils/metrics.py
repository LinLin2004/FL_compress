import torch

class TopKAccuracy:
    """
    Top-K accuracy metric class.
    """
    
    def __init__(self, k: int = 5):
        """
        Initialize Top-K accuracy metric.
        
        Args:
            k (int): The K value for top-k accuracy
        """
        self.k = k
        self.__name__ = f'top{k}_accuracy'
    
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Top-K accuracy for a single batch.
        
        Args:
            outputs (torch.Tensor): Model outputs, shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels, shape (batch_size,)
            
        Returns:
            torch.Tensor: Top-K accuracy as a scalar tensor
        """
        with torch.no_grad():
            # Get top-k predictions
            if outputs.dim() == 1: # for binary classification
                ne_outputs = 1 - outputs
                outputs = torch.stack((ne_outputs, outputs), dim=1)

            _, top_k_pred = torch.topk(outputs, self.k, dim=1)
            
            # Expand targets to match top_k_pred shape
            targets_expanded = targets.unsqueeze(1).expand_as(top_k_pred)
            
            # Check if target is in top-k predictions
            correct = (top_k_pred == targets_expanded).any(dim=1).float()
            
            return correct.mean()