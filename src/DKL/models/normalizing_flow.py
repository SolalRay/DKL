import torch
import torch.nn as nn

def init_zero(layer):
    """Function to initialize the weights and biases of a layer to zero, so that the flow starts as an identity transform."""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    # Defines an MLP for calculating s (scale) and t (translation)
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # First dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Second dropout
            nn.Linear(hidden_dim, output_dim)
        )
        # We force zero weights for the last layer to ensure identity initialization
        init_zero(self.net[-1]) 

    def forward(self, x):
        return self.net(x)

class AffineCoupling(nn.Module):
    # Performs a RealNVP-type affine coupling layer
    def __init__(self, input_dim=2):
        super().__init__()
        self.s_net1 = MLP(input_dim // 2, input_dim // 2)
        self.t_net1 = MLP(input_dim // 2, input_dim // 2)
        self.s_net2 = MLP(input_dim // 2, input_dim // 2)
        self.t_net2 = MLP(input_dim // 2, input_dim // 2)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # Coupling for x2 based on x1
        s1 = self.s_net1(x1)
        t1 = self.t_net1(x1)
        z2 = x2 * torch.exp(s1) + t1
        
        # Coupling for x1 based on updated z2
        s2 = self.s_net2(z2)
        t2 = self.t_net2(z2)
        z1 = x1 * torch.exp(s2) + t2
        
        z = torch.cat([z1, z2], dim=-1)
        return z

class Permutation(nn.Module):
    # Permutes input dimensions
    def __init__(self, dims=2):
        super().__init__()
    def forward(self, x):
        return torch.flip(x, dims=[-1])

class RealNVP(nn.Module):
    # Implements a simplified RealNVP
    def __init__(self, num_blocks=3):
        super().__init__()
        modules = []
        for _ in range(num_blocks):
            modules.append(AffineCoupling(input_dim=2))
            modules.append(Permutation(dims=2))
        self.flow = nn.Sequential(*modules)
        print(f"RealNVP model created with {num_blocks} affine blocks.")
        print("MLP output layers initialized to zeros for near-identity transform.")

    def forward(self, x):
        return self.flow(x)