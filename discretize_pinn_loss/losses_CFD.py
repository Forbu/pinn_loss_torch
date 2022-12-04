from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data
import torch

import math
from torch_geometric.nn import MetaLayer

from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator, SpatialSecondDerivativeOperator

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

class CompressibleFluidLoss(nn.Module):
    """
    This loss function computes the divergence of the velocity field and the laplacian of the velocity field
    """
    def __init__(self, index_node_x=0, index_node_y=1, index_edge_x=0, index_edge_y=1, delta_x=1./512):
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

    def forward(self, graph_v, graph_v_previous, graph_p, graph_p_previous, graph_rho, graph_rho_previous, M, eta, zeta, dt):
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

        graph_tmp = Data(x=graph_v.x * graph_p.x, edge_index=edge_index_x, edge_attr=edge_attr_x)

        # compute the derivative in x direction
        derivative_x = self.spatial_derivative_operator_xx(graph_tmp)

        # get the input node and reshape it because the model expects a batch
        bool_index = (edge_attr[:, self.index_node_y] != 0)
        edge_attr_y = edge_attr[bool_index, :]
        edge_index_y = graph_v.edge_index[:, bool_index]

        graph_tmp = Data(x=graph_v.x * graph_p.x, edge_index=edge_index_y, edge_attr=edge_attr_y)

        # compute the derivative in y direction
        derivative_y = self.spatial_derivative_operator_yy(graph_tmp)

        loss_continuity = derivative_x + derivative_y + (graph_p.x - graph_p_previous.x)/ dt

        

        return loss_continuity

