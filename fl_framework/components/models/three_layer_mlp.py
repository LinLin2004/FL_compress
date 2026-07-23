import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ThreeLayerMLP(nn.Module):
    def __init__(self,in_dim=3*32*32):
        super(ThreeLayerMLP,self).__init__()
        self.lin1 = nn.Linear(in_dim,128)
        self.lin2 = nn.Linear(128,64)
        self.lin3 = nn.Linear(64,10)
        self.relu = nn.ReLU()
        init.xavier_uniform_(self.lin1.weight)
        init.xavier_uniform_(self.lin2.weight)
        init.xavier_uniform_(self.lin3.weight)

    def forward(self,x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        return x