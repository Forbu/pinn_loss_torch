from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data
import torch

import math
from torch_geometric.nn import MetaLayer

"""
In this file we define the PINN model and the loss function for Inhomogenous, incompressible Navier-Stokes

The equation we want to solve is:
    div(v) = 0
    v_t + v * grad(v) = -grad(p) + mu * laplacian(v)

With the following boundary conditions:
    v(x, t) = 0 on the boundary

    Also we can directly compute the pressure from the velocity:
    p = -div(v)


We have a constant viscosity mu and a constant density rho

 



"""