"""
Here we will define the loss operator for the discretize pinn
Basicly we implement the temporal derivative and the spatial derivative
and them we construct the different specific loss operator (burger loss, ...)
"""

from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum