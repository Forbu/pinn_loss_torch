
from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data
import torch

import math

from discretize_pinn_loss.loss_operator import TemporalDerivativeOperator, SpatialBurgerDerivativeOperator, SpatialSecondDerivativeOperator

class BurgerDissipativeLossOperator(Module):
    """
    This class is used to compute the dissipative loss of the burger equation
    """
    def __init__(self, index_derivative_node, index_derivative_edge, delta_t, mu) -> None:
        super().__init__()

        self.index_derivative_node = index_derivative_node
        self.index_derivative_edge = index_derivative_edge

        self.delta_t = delta_t
        self.mu = mu

        self.temporal_derivative_operator = TemporalDerivativeOperator(self.index_derivative_node, self.delta_t)
        self.spatial_derivative_operator = SpatialBurgerDerivativeOperator(self.index_derivative_node, self.index_derivative_edge)

    def forward(self, graph_t, graph_t_1, mask=None):
        """
        TODO care about dim_node
        """
        # compute the temporal derivative
        temporal_derivative = self.temporal_derivative_operator(graph_t, graph_t_1)

        # compute the spatial derivative
        spatial_derivative = self.spatial_derivative_operator(graph_t_1)

        graph_first_order = Data(x=spatial_derivative.unsqueeze(-1), edge_index=graph_t_1.edge_index, edge_attr=graph_t_1.edge_attr)

        # second order derivative
        second_order_derivative = self.spatial_derivative_operator(graph_first_order)

        # compute the loss
        loss = temporal_derivative + spatial_derivative * graph_t_1.x[:, self.index_derivative_node] - self.mu * second_order_derivative

        if mask is not None:
            loss = loss * mask.squeeze()

        return loss


class BurgerDissipativeImplicitLossOperator(Module):
    """
    This class is used to compute the dissipative loss of the burger equation
    """
    def __init__(self, index_derivative_node, index_derivative_edge, delta_t, delta_x, mu, index_limit=2, index_mask=1) -> None:
        super().__init__()

        self.index_derivative_node = index_derivative_node
        self.index_derivative_edge = index_derivative_edge
        self.index_limit = index_limit
        self.index_mask = index_mask

        self.delta_t = delta_t
        self.delta_x = delta_x
        self.mu = mu

        self.temporal_derivative_operator = TemporalDerivativeOperator(self.index_derivative_node, self.delta_t)
        self.spatial_derivative_operator = SpatialBurgerDerivativeOperator(self.index_derivative_node, self.index_derivative_edge)
        self.spatial_secondderivative_operator = SpatialSecondDerivativeOperator(self.index_derivative_node, self.index_derivative_edge, self.delta_x)

    def forward(self, graph_t, graph_t_1, mask=None):
        """
        mask is used to mask the loss on the boundary
        """
        
        #graph_t.x[:, self.index_derivative_node] = torch.where(graph_t_1.x[:, self.index_mask] == 0, graph_t_1.x[:, self.index_limit], graph_t.x[:, self.index_derivative_node])

        # compute the temporal derivative
        temporal_derivative = self.temporal_derivative_operator(graph_t, graph_t_1)

        # compute the spatial derivative
        spatial_derivative = self.spatial_derivative_operator(graph_t)

        # second order derivative
        second_order_derivative = self.spatial_secondderivative_operator(graph_t)

        # compute the loss
        loss = temporal_derivative + spatial_derivative * graph_t.x[:, self.index_derivative_node] - self.mu / math.pi * second_order_derivative

        if mask is not None:
            loss = loss * mask.squeeze()

        return loss


class BurgerDissipativeMixLossOperator(Module):
    """
    This class is used to compute the dissipative loss of the burger equation
    """
    def __init__(self, index_derivative_node, index_derivative_edge, delta_t, delta_x, mu, index_limit=2, index_mask=1) -> None:
        super().__init__()

        self.index_derivative_node = index_derivative_node
        self.index_derivative_edge = index_derivative_edge
        self.index_limit = index_limit
        self.index_mask = index_mask

        self.delta_t = delta_t
        self.delta_x = delta_x
        self.mu = mu

        self.temporal_derivative_operator = TemporalDerivativeOperator(self.index_derivative_node, self.delta_t)
        self.spatial_derivative_operator = SpatialBurgerDerivativeOperator(self.index_derivative_node, self.index_derivative_edge)
        self.spatial_secondderivative_operator = SpatialSecondDerivativeOperator(self.index_derivative_node, self.index_derivative_edge, self.delta_x)

    def forward(self, graph_t, graph_t_1, mask=None):
        """
        mask is used to mask the loss on the boundary
        """

        # compute the temporal derivative
        temporal_derivative = self.temporal_derivative_operator(graph_t, graph_t_1)

        # compute the spatial derivative
        spatial_derivative = self.spatial_derivative_operator(graph_t)

        # second order derivative
        second_order_derivative = self.spatial_secondderivative_operator(graph_t)

        # compute the spatial derivative
        spatial_derivative_init = self.spatial_derivative_operator(graph_t_1)

        # second order derivative
        second_order_derivative_init = self.spatial_secondderivative_operator(graph_t_1)

        # compute the loss
        loss = temporal_derivative +  (spatial_derivative * graph_t.x[:, self.index_derivative_node] - self.mu / math.pi * second_order_derivative) + \
                            (spatial_derivative_init * graph_t_1.x[:, self.index_derivative_node] - self.mu / math.pi * second_order_derivative_init)

        if mask is not None:
            loss = loss * mask.squeeze()

        return loss