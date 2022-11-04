"""
Here we will define the loss operator for the discretize pinn
Basicly we implement the temporal derivative and the spatial derivative
and them we construct the different specific loss operator (burger loss, ...)
"""

from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean

class EdgeSpatialDerivative(Module):
    """
    This class is used to compute the derivative of the edge
    """
    def __init__(self, dim):
        super(EdgeSpatialDerivative, self).__init__()
        self.dim = dim

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
    def __init__(self, dim):
        super(NodeSpatialDerivative, self).__init__()
        self.dim = dim

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the node
        It is the mean of the derivative of the edge
        """
        derivative = scatter_mean(edge_attr, edge_index[1], dim=0)
        return derivative


# here we define the loss operator
class SpatialDerivativeOperator(Module):

    def __init__(self, dim_node, dim_edge) -> None:
        super().__init__()

        self.dim_node = dim_node
        self.dim_edge = dim_edge

        # we define the meta layer TODO CHANGE
        self.meta_layer = MetaLayer(
            edge_model=EdgeSpatialDerivative(),
            node_model=NodeSpatialDerivative(),)

    def forward(self, graph):
        """
        TODO care about dim_node and dim_edge
        """
        return self.meta_layer(graph.x, graph.edge_index, graph.edge_attr)

class TemporalDerivativeOperator(Module):
    """
    This class is used to compute the temporal derivative of the graph
    """
    def __init__(self, dim_node, dim_edge) -> None:
        super().__init__()

        self.dim_node = dim_node
        self.dim_edge = dim_edge

    def forward(self, graph_t, graph_t_1):
        """
        TODO care about dim_node and dim_edge
        """
        return graph_t.x - graph_t_1.x