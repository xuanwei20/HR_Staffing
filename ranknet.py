import torch.nn as nn

class RankNetModel(nn.Module):
    """Neural network for pairwise ranking"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = [] 
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim)) # Fully connected layer
            layers.append(nn.ReLU()) # Activation function, keeps positive values, sets negative to 0
            layers.append(nn.Dropout(0.2)) # Regularization
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)