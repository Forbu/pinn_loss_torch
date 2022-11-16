"""
Here we will define the loss operator for the discretize pinn
Basicly we implement the temporal derivative and the spatial derivative
and them we construct the different specific loss operator (burger loss, ...)
"""

from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data
import torch

import math

class EdgeSpatialDerivative(Module):
    """
    This class is used to compute the derivative of the edge
    """
    def __init__(self):
        super(EdgeSpatialDerivative, self).__init__()

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the edge
        """
        local_derivative = (dest - src) / edge_attr
        return local_derivative

class NodeSpatialDerivative(Module):
    """
    This class is used to compute the derivative of the node
    """
    def __init__(self):
        super(NodeSpatialDerivative, self).__init__()

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the node
        It is the mean of the derivative of the edge
        """
        nb_node = x.shape[0]

        derivative = scatter_mean(edge_attr, edge_index[1], dim=0, dim_size=nb_node)
        return derivative

class EdgeSpatialSecondDerivative(Module):
    """
    This class is used to compute the derivative of the edge
    """
    def __init__(self):
        super(EdgeSpatialSecondDerivative, self).__init__()

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the edge
        """
        local_info = src 
        return local_info

class NodeSpatialSecondDerivative(Module):
    """
    This class is used to compute the derivative of the node
    """
    def __init__(self, delta_x=None):
        super(NodeSpatialSecondDerivative, self).__init__()
        self.delta_x = delta_x

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the node
        It is the mean of the derivative of the edge
        """
        nb_node = x.shape[0]
        sum_extrem = scatter_sum(edge_attr, edge_index[1], dim=0, dim_size=nb_node)

        derivative = (sum_extrem - 2 * x) / (self.delta_x)**2

        return derivative

# here we define the loss operator
class SpatialDerivativeOperator(Module):

    def __init__(self, index_derivative_node, index_derivative_edge) -> None:
        super().__init__()

        self.index_derivative_node = index_derivative_node
        self.index_derivative_edge = index_derivative_edge

        # we define the meta layer TODO CHANGE
        self.meta_layer = MetaLayer(
            edge_model=EdgeSpatialDerivative(),
            node_model=NodeSpatialDerivative(),)

    def forward(self, graph):
        """
        Care about dim_node and dim_edge
        """
        nodes_input = graph.x[:, self.index_derivative_node]
        edges_input = graph.edge_attr[:, self.index_derivative_edge]
        
        nodes, _, _ = self.meta_layer(nodes_input, graph.edge_index, edges_input)

        return nodes

# here we define the loss operator
class SpatialSecondDerivativeOperator(Module):

    def __init__(self, index_derivative_node, index_derivative_edge, delta_x) -> None:
        super().__init__()

        self.index_derivative_node = index_derivative_node
        self.index_derivative_edge = index_derivative_edge

        # we define the meta layer TODO CHANGE
        self.meta_layer = MetaLayer(
            edge_model=EdgeSpatialSecondDerivative(),
            node_model=NodeSpatialSecondDerivative(delta_x),)

    def forward(self, graph):
        """
        Care about dim_node and dim_edge
        """
        nodes_input = graph.x[:, self.index_derivative_node]
        edges_input = graph.edge_attr[:, self.index_derivative_edge]
        
        nodes, _, _ = self.meta_layer(nodes_input, graph.edge_index, edges_input)

        return nodes


class TemporalDerivativeOperator(Module):
    """
    This class is used to compute the temporal derivative of the graph
    """
    def __init__(self, index_derivator, delta_t) -> None:
        super().__init__()

        self.index_derivator = index_derivator
        self.delta_t = delta_t

    def forward(self, graph_t, graph_t_1):
        """
        TODO care about dim_node
        """
        return (graph_t.x[:, self.index_derivator] - graph_t_1.x[:, self.index_derivator])/self.delta_t

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
        self.spatial_derivative_operator = SpatialDerivativeOperator(self.index_derivative_node, self.index_derivative_edge)

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
        self.spatial_derivative_operator = SpatialDerivativeOperator(self.index_derivative_node, self.index_derivative_edge)
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
        loss = temporal_derivative + spatial_derivative * graph_t.x[:, self.index_derivative_node] - self.mu * second_order_derivative

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
        self.spatial_derivative_operator = SpatialDerivativeOperator(self.index_derivative_node, self.index_derivative_edge)
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
        loss = temporal_derivative + 1/2 * (spatial_derivative * graph_t.x[:, self.index_derivative_node] - self.mu / math.pi * second_order_derivative) + \
                            1/2 * (spatial_derivative_init * graph_t_1.x[:, self.index_derivative_node] - self.mu / math.pi * second_order_derivative_init)

        if mask is not None:
            loss = loss * mask.squeeze()

        return loss