import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    #def __init__(self, state_size, action_size, seed):
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        hidden_layer_size = [64,64]
        layer_size = [state_size] + hidden_layer_size + [action_size]
        self.layers = nn.ModuleList([nn.Linear(layer_size[i],layer_size[i+1]) for i in range(len(layer_size)-1)])

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.layers[0](state)
        for layer in self.layers[1:]: x = layer(F.relu(x))
        return x