import torch
import torch.nn as nn

class SingleLayerPerceptron(nn.Module):
    """
    Inputs                Linear/Function             Output
    [batch, 1, 28, 28] -> Linear(28*28, 32)       -> [batch, 32]  # Hidden layer
                       -> Activation (ReLU)       -> [batch, 32]
                       -> Linear(32, 10)          -> [batch, 10]  # Classification layer
                       -> Softmax (optional)      -> [batch, 10]  # Probability output
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleLayerPerceptron, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.classification_layer = nn.Linear(hidden_size, output_size)
        # self.classification_layer = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # Optional

    def forward(self, x):
        # Flatten input: [batch_size, 1, 28, 28] -> [batch_size, 784]
        out = x.view(x.size(0), -1)
        out = self.hidden_layer(out)
        out = self.activation(out)
        out = self.classification_layer(out)
        out = self.softmax(out)  # Optional: keep if using NLLLoss or for probability output
        return out
