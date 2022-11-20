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


#################### Loss Operator #####################
############### Classic derivative #####################
########################################################

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

        # return None if edge_attr is 0
        local_derivative[edge_attr == 0] = -99999

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

        # delete edge_attr == -99999
        edge_index = edge_index[:, edge_attr != -99999]
        edge_attr = edge_attr[edge_attr != -99999]

        derivative = scatter_mean(edge_attr, edge_index[1], dim=0, dim_size=nb_node)
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

#################### Loss Operator #####################
############### Burger derivative ######################
########################################################

class EdgeSpatialUpWindDerivative(Module):
    """
    This class is used to compute the derivative of the edge
    """
    def __init__(self):
        super(EdgeSpatialUpWindDerivative, self).__init__()

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the edge

        4 cases :
            - u > 0 and edge_attrib > 0 => local_derivative
            - u < 0 and edge_attrib > 0 => 0
            - u > 0 and edge_attrib < 0 => 0
            - u < 0 and edge_attrib < 0 => local_derivative
        """
        local_derivative = torch.where(src*edge_attr > 0 , (dest - src) / edge_attr, 0)

        return local_derivative



class NodeSpatialBurgerDerivative(Module):
    """
    This class is used to compute the derivative of the node
    This is the upwind version !
    """
    def __init__(self):
        super(NodeSpatialBurgerDerivative, self).__init__()

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the node
        It is the mean of the derivative of the edge
        """
        nb_node = x.shape[0]

        derivative = scatter_sum(edge_attr, edge_index[1], dim=0, dim_size=nb_node)
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
class SpatialBurgerDerivativeOperator(Module):

    def __init__(self, index_derivative_node, index_derivative_edge) -> None:
        super().__init__()

        self.index_derivative_node = index_derivative_node
        self.index_derivative_edge = index_derivative_edge

        # we define the meta layer TODO CHANGE
        self.meta_layer = MetaLayer(
            edge_model=EdgeSpatialUpWindDerivative(),
            node_model=NodeSpatialBurgerDerivative(),)

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

#################### Loss Operator #####################
############### Temporal derivative ####################
########################################################


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

