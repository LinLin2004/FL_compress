import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, feature_size=22):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(feature_size + 1, dtype=torch.float64), requires_grad=True)
        torch.nn.init.normal_(self.w)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, feature_size], input for network
        Returns:
            out: [batch_size, 1], output from network
        """
        # Add bias term
        x = torch.cat([x, torch.ones((x.shape[0], 1), dtype=torch.float64, device=x.device)], dim=1)
        out = x @ self.w
        return torch.sigmoid(out)
