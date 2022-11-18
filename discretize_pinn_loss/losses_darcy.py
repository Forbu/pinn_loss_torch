from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data
import torch

import math

from discretize_pinn_loss.loss_operator import TemporalDerivativeOperator, SpatialDerivativeOperator

class Nabla2DOperator(Module):
    def __init__(self, delta_x, delta_y) -> None:
        super().__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.derivative_x = SpatialDerivativeOperator(index_derivative_node=0, index_derivative_edge=0)
        self.derivative_y = SpatialDerivativeOperator(index_derivative_node=0, index_derivative_edge=1)

    def forward(self, graph):
        
        # we compute the derivative of the x component
        derivative_x = self.derivative_x(graph)

        # we compute the derivative of the y component
        derivative_y = self.derivative_y(graph)

        # we compute the nabla2d
        nabla2d = derivative_x + derivative_y

        return nabla2d

## we define the operator for the darcy loss equation
class DarcyFlowOperator(Module):
    """
    Class that define the operator for the darcy flow equation
    """
    def __init__(self, delta_x, delta_y) -> None:
        super().__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.nabla2d_operator = Nabla2DOperator(delta_x=delta_x, delta_y=delta_y)

    def forward(self, out, a_x, f, mask=None):

        # first we compute the nabla of the out
        nabla2d_out = self.nabla2d_operator(out)

        tmp_flow = a_x * nabla2d_out

        # we compute the nabla of the tmp_flow
        nabla2d_tmp_flow = self.nabla2d_operator(tmp_flow)

        pde_loss = nabla2d_tmp_flow - f

        if mask is not None:
            pde_loss = pde_loss * mask

        return pde_loss


