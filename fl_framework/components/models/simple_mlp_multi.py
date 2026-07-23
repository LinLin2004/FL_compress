import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Inputs                Linear/Function        Output
    [128, 1, 28, 28]   -> Linear(28*28, 100) -> [128, 100]  # first hidden layer
                       -> Tanh               -> [128, 100]  # Tanh activation function, may sigmoid
                       -> Linear(100, 100)   -> [128, 100]  # third hidden layer
                       -> Tanh               -> [128, 100]  # Tanh activation function, may sigmoid
                       -> Linear(100, 10)    -> [128, 10]   # Classification Layer                                                          
   """
    def __init__(self, input_size, hidden_size, output_size, SEED=100):
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.classification_layer = torch.nn.Linear(hidden_size, output_size)
        
        self.tanh1 = torch.nn.Tanh()
        self.tanh2 = torch.nn.Tanh()
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        """Defines the computation performed at every call.
           Should be overridden by all subclasses.
        Args:
            x: [batch_size, channel, height, width], input for network
        Returns:
            out: [batch_size, n_classes], output from network
        """
        
        out = x.view(x.size(0), -1) # flatten x in [128, 784]
        out = self.tanh1(out)
        out = self.hidden(out)
        out = self.tanh2(out)
        out = self.classification_layer(out)
        out = self.softmax(out)
        return out