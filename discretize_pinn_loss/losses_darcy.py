from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data
import torch

import math

from discretize_pinn_loss.loss_operator import TemporalDerivativeOperator, SpatialDerivativeOperator

class Nabla2DOperator(Module):
    def __init__(self, delta_x, delta_y, index_derivative_node=0,
                                             index_derivative_x=0, index_derivative_y=1) -> None:
        super().__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.derivative_x = SpatialDerivativeOperator(index_derivative_node=index_derivative_node, index_derivative_edge=index_derivative_x)
        self.derivative_y = SpatialDerivativeOperator(index_derivative_node=index_derivative_node, index_derivative_edge=index_derivative_y)

    def forward(self, graph):
        
        # we compute the derivative of the x component
        derivative_x = self.derivative_x(graph)

        # we compute the derivative of the y component
        derivative_y = self.derivative_y(graph)

        # we compute the nabla2d
        nabla2d = torch.cat((derivative_x.reshape((-1, 1)), derivative_y.reshape((-1, 1))), dim=1)

        return nabla2d

class Nabla2DProductOperator(Module):
    def __init__(self, delta_x, delta_y, index_derivative_node=0,
                                             index_derivative_x=0, index_derivative_y=1) -> None:
        super().__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.derivative_x = SpatialDerivativeOperator(index_derivative_node=index_derivative_node, index_derivative_edge=index_derivative_x)
        self.derivative_y = SpatialDerivativeOperator(index_derivative_node=index_derivative_node, index_derivative_edge=index_derivative_y)

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
    def __init__(self, index_derivative_node=0, index_derivative_x=0, index_derivative_y=1, delta_x=0.1, delta_y=0.1) -> None:
        super().__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.index_derivative_node = index_derivative_node
        self.index_derivative_x = index_derivative_x
        self.index_derivative_y = index_derivative_y

        self.nabla2d_operator = Nabla2DOperator(delta_x=delta_x, delta_y=delta_y, index_derivative_node=index_derivative_node,
                                             index_derivative_x=index_derivative_x, index_derivative_y=index_derivative_y)
        self.nabla2d_product_operator = Nabla2DProductOperator(delta_x=delta_x, delta_y=delta_y,
                                                index_derivative_node=index_derivative_node, index_derivative_x=index_derivative_x, index_derivative_y=index_derivative_y)

    def forward(self, out, a_x, f=1, mask=None):
        """
        out and a_x aare graphs
        out is the output of the neural network (FNO)
        a_x is the input of the neural network (FNO)
        """

        # first we compute the nabla of the out
        nabla2d_out = self.nabla2d_operator(out) # shape (nb_node, 2)

        tmp_flow = a_x.x[:, [self.index_derivative_node]] * nabla2d_out # shape (nb_node, 2)

        # create the graph
        tmp_flow_graph = Data(x=tmp_flow, edge_index=out.edge_index, edge_attr=out.edge_attr)

        # we compute the nabla of the tmp_flow
        nabla2d_tmp_flow = self.nabla2d_product_operator(tmp_flow_graph) # shape (nb_node, 1)

        pde_loss = nabla2d_tmp_flow - f

        if mask is not None:
            pde_loss = pde_loss * mask

        return pde_loss


