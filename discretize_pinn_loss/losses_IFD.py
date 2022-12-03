from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data
import torch

import math
from torch_geometric.nn import MetaLayer

from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator, SpatialSecondDerivativeOperator

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

class DivLoss(nn.Module):
    """
    This loss function computes the divergence of the velocity field
    """
    def __init__(self, index_node_x=0, index_node_y=1, index_edge_x=0, index_edge_y=1):
        super().__init__()
        
        self.index_edge_x = index_edge_x
        self.index_edge_y = index_edge_y

        self.index_node_x = index_node_x
        self.index_node_y = index_node_y

        self.spatial_derivative_operator_x = SpatialDerivativeOperator(index_derivative_node=index_node_x, index_derivative_edge=index_edge_x)
        self.spatial_derivative_operator_y = SpatialDerivativeOperator(index_derivative_node=index_node_y, index_derivative_edge=index_edge_y)


    def forward(self, graph_v):
        """
        :param v: the velocity field
        :return: the divergence of the velocity field
        """

        edge_attr = graph_v.edge_attr

        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_x] != 0)
        edge_attr_x = edge_attr[bool_index, self.index_edge_x]
        edge_index_x = graph_v.edge_index[:, bool_index]

        # compute the derivative in x direction
        derivative_x = self.spatial_derivative_operator_x(edge_attr_x, edge_index_x, graph_v.x)

        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_y] != 0)
        edge_attr_y = edge_attr[bool_index, self.index_edge_y]
        edge_index_y = graph_v.edge_index[:, bool_index]

        # compute the derivative in y direction
        derivative_y = self.spatial_derivative_operator_y(edge_attr_y, edge_index_y, graph_v.x)

        # compute the divergence
        divergence = derivative_x + derivative_y

        return divergence

class LaplacianVectorLoss(nn.Module):
    """
    This loss function computes the laplacian of the velocity field
    """
    def __init__(self, index_node_x=0, index_node_y=1, index_edge_x=0, index_edge_y=1, delta_x=0.01):
        super().__init__()

        self.index_edge_x = index_edge_x
        self.index_edge_y = index_edge_y

        self.index_node_x = index_node_x
        self.index_node_y = index_node_y

        self.delta_x = delta_x

        self.spatial_second_derivative_operator_xx = SpatialSecondDerivativeOperator(index_derivative_node=index_node_x,
                                             index_derivative_edge=index_edge_x, delta_x=delta_x)

        self.spatial_second_derivative_operator_yy = SpatialSecondDerivativeOperator(index_derivative_node=index_node_y,
                                                index_derivative_edge=index_edge_y, delta_x=delta_x)

        self.spatial_second_derivative_operator_xy = SpatialSecondDerivativeOperator(index_derivative_node=index_node_x,
                                                index_derivative_edge=index_edge_y, delta_x=delta_x)

        self.spatial_second_derivative_operator_yx = SpatialSecondDerivativeOperator(index_derivative_node=index_node_y,
                                                index_derivative_edge=index_edge_x, delta_x=delta_x)


    def forward(self, graph_v):
        """
        :param v: the velocity field
        :return: the laplacian of the velocity field
        """
        # here we want to compute the laplacian of the velocity field
        # to do that we have to compute the second derivative in x and y direction

        edge_attr = graph_v.edge_attr

        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_x] != 0)
        edge_attr_x = edge_attr[bool_index, :]
        edge_index_x = graph_v.edge_index[:, bool_index]

        graph_tmp = Data(x=graph_v.x, edge_index=edge_index_x, edge_attr=edge_attr_x)

        # compute the second derivative in x direction
        second_derivative_x = self.spatial_second_derivative_operator_xx(graph_tmp)
        second_derivative_xy = self.spatial_second_derivative_operator_yx(graph_tmp)


        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_y] != 0)
        edge_attr_y = edge_attr[bool_index, :]
        edge_index_y = graph_v.edge_index[:, bool_index]

        graph_tmp = Data(x=graph_v.x, edge_index=edge_index_y, edge_attr=edge_attr_y)

        # compute the second derivative in y direction
        second_derivative_y = self.spatial_second_derivative_operator_yy(graph_tmp)
        second_derivative_yx = self.spatial_second_derivative_operator_xy(graph_tmp)
        
        return second_derivative_x, second_derivative_xy, second_derivative_y, second_derivative_yx 

class IncompressibleFluidLoss(nn.Module):
    """
    This loss function computes the divergence of the velocity field and the laplacian of the velocity field
    """
    def __init__(self, index_node_x=0, index_node_y=1, index_edge_x=0, index_edge_y=1, delta_x=0.01):
        super().__init__()

        self.index_edge_x = index_edge_x
        self.index_edge_y = index_edge_y

        self.index_node_x = index_node_x
        self.index_node_y = index_node_y

        self.spatial_derivative_operator_xx = SpatialDerivativeOperator(index_derivative_node=index_node_x, index_derivative_edge=index_edge_x)
        self.spatial_derivative_operator_yy = SpatialDerivativeOperator(index_derivative_node=index_node_y, index_derivative_edge=index_edge_y)

        self.spatial_derivative_operator_xy = SpatialDerivativeOperator(index_derivative_node=index_node_x, index_derivative_edge=index_edge_y)
        self.spatial_derivative_operator_yx = SpatialDerivativeOperator(index_derivative_node=index_node_y, index_derivative_edge=index_edge_x)

        self.laplacian_loss = LaplacianVectorLoss(index_node_x, index_node_y, index_edge_x, index_edge_y, delta_x)

    def forward(self, graph_v, graph_v_previous, graph_p, mu, dt, force):
        """
        :param v: the velocity field
        :param p: the pressure field
        :param mu: the viscosity
        :param rho: the density
        :param dt: the time step
        :param force: the force
        :return: the divergence of the velocity field and the laplacian of the velocity field
        """
        
        # compute derivative in x direction and y direction
        # first we have to get the edge attributes and the edge indices
        edge_attr = graph_v.edge_attr

        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_x] != 0)
        edge_attr_x = edge_attr[bool_index, :]
        edge_index_x = graph_v.edge_index[:, bool_index]

        graph_tmp = Data(x=graph_v.x, edge_index=edge_index_x, edge_attr=edge_attr_x)

        # compute the derivative in x direction
        derivative_x = self.spatial_derivative_operator_xx(graph_tmp)

        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_y] != 0)
        edge_attr_y = edge_attr[bool_index, :]
        edge_index_y = graph_v.edge_index[:, bool_index]

        graph_tmp = Data(x=graph_v.x, edge_index=edge_index_y, edge_attr=edge_attr_y)

        # compute the derivative in y direction
        derivative_y = self.spatial_derivative_operator_yy(graph_tmp)

        # compute the divergence THIS IS THE FIRST PART OF THE LOSS
        divergence = derivative_x + derivative_y

        # now we compute the cross derivative (xy and yx)
        # first we have to get the edge attributes and the edge indices
        edge_attr = graph_v.edge_attr

        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_y] != 0)
        edge_attr_y = edge_attr[bool_index, :]
        edge_index_y = graph_v.edge_index[:, bool_index]

        graph_tmp = Data(x=graph_v.x, edge_index=edge_index_y, edge_attr=edge_attr_y)

        # compute the derivative in x direction
        derivative_xy = self.spatial_derivative_operator_xy(graph_tmp)

        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_x] != 0)
        edge_attr_x = edge_attr[bool_index, :]
        edge_index_x = graph_v.edge_index[:, bool_index]

        graph_tmp = Data(x=graph_v.x, edge_index=edge_index_x, edge_attr=edge_attr_x)

        # compute the derivative in y direction
        derivative_yx = self.spatial_derivative_operator_yx(graph_tmp)
        
        # compute the laplacian
        second_derivative_x, second_derivative_xy, second_derivative_y, second_derivative_yx =\
                                                                         self.laplacian_loss(graph_v)

        # compute the divergence THIS IS THE SECOND PART OF THE LOSS
        loss_momentum_x = ( (graph_v.x[:, 0] - graph_v_previous.x[:, 0]) / dt + graph_v.x[:, 0] * derivative_x + \
                                                                             graph_v.x[:, 1] * derivative_xy) - \
                            mu * (second_derivative_x + second_derivative_yx) - force[:, 0]  
        
        loss_momentum_y = ( (graph_v.x[:, 1] - graph_v_previous.x[:, 1]) / dt + graph_v.x[:, 0] * derivative_yx + \
                                                                                graph_v.x[:, 1] * derivative_y) - \
                            mu * (second_derivative_y + second_derivative_xy) - force[:, 1]


        print("loss_momentum_x", loss_momentum_x)
        print("loss_momentum_y", loss_momentum_y)

        # compute the divergence THIS IS THE THIRD PART OF THE LOSS
        loss_continuity = divergence

        return loss_momentum_x, loss_momentum_y, loss_continuity

