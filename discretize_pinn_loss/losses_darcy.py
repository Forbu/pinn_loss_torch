from torch.nn import Module        
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import Data
import torch

import math
from torch_geometric.nn import MetaLayer

from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator

class Nabla2DOperator(Module):
    def __init__(self, delta_x, delta_y, index_derivative_node=0,
                                             index_derivative_x=0, index_derivative_y=1) -> None:
        super().__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.index_derivative_node = index_derivative_node
        self.index_derivative_x = index_derivative_x
        self.index_derivative_y = index_derivative_y

        self.derivative_x = SpatialDerivativeOperator(index_derivative_node=index_derivative_node, index_derivative_edge=index_derivative_x)
        self.derivative_y = SpatialDerivativeOperator(index_derivative_node=index_derivative_node, index_derivative_edge=index_derivative_y)

    def forward(self, graph):

        # we remove edge where edge_attr is 0
        edge_index_x = graph.edge_index[:, graph.edge_attr[:, self.index_derivative_x] != 0]
        edge_attr_x = graph.edge_attr[graph.edge_attr[:, self.index_derivative_x] != 0]

        edge_index_y = graph.edge_index[:, graph.edge_attr[:, self.index_derivative_y] != 0]
        edge_attr_y = graph.edge_attr[graph.edge_attr[:, self.index_derivative_y] != 0]

        # we create new graph for x and y
        graph_x = Data(x=graph.x, edge_index=edge_index_x, edge_attr=edge_attr_x)
        graph_y = Data(x=graph.x, edge_index=edge_index_y, edge_attr=edge_attr_y)
            
        # we compute the derivative of the x component
        derivative_x = self.derivative_x(graph_x)

        # we compute the derivative of the y component
        derivative_y = self.derivative_y(graph_y)

        # we compute the nabla2d
        nabla2d = torch.cat((derivative_x.reshape((-1, 1)), derivative_y.reshape((-1, 1))), dim=1)

        return nabla2d

class Nabla2DProductOperator(Module):
    def __init__(self, delta_x, delta_y, index_derivative_node=0,
                                             index_derivative_x=0, index_derivative_y=1) -> None:
        super().__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.index_derivative_node = index_derivative_node
        self.index_derivative_x = index_derivative_x
        self.index_derivative_y = index_derivative_y

        self.derivative_x = SpatialDerivativeOperator(index_derivative_node=index_derivative_x, index_derivative_edge=index_derivative_x)
        self.derivative_y = SpatialDerivativeOperator(index_derivative_node=index_derivative_y, index_derivative_edge=index_derivative_y)

    def forward(self, graph):

        # we remove edge where edge_attr is 0
        edge_index_x = graph.edge_index[:, graph.edge_attr[:, self.index_derivative_x] != 0]
        edge_attr_x = graph.edge_attr[graph.edge_attr[:, self.index_derivative_x] != 0]

        edge_index_y = graph.edge_index[:, graph.edge_attr[:, self.index_derivative_y] != 0]
        edge_attr_y = graph.edge_attr[graph.edge_attr[:, self.index_derivative_y] != 0]

        # we create new graph for x and y
        graph_x = Data(x=graph.x, edge_index=edge_index_x, edge_attr=edge_attr_x)
        graph_y = Data(x=graph.x, edge_index=edge_index_y, edge_attr=edge_attr_y)
        
        # we compute the derivative of the x component
        derivative_x = self.derivative_x(graph_x)

        # we compute the derivative of the y component
        derivative_y = self.derivative_y(graph_y)

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

    def forward(self, out, a_x, f=1., mask=None):
        """
        out and a_x are graphs
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

        pde_loss = nabla2d_tmp_flow + f

        if mask is not None:
            pde_loss = pde_loss * (1 - mask)

        return pde_loss

############################################################################################################
############################### we define the new loss function ############################################
############################################################################################################

class EdgeSpatialAverage(Module):
    """
    This class is used to compute the derivative of the edge
    """
    def __init__(self):
        super(EdgeSpatialAverage, self).__init__()

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the edge
        """
        local_info = src 
        return local_info

class NodeSpatialAverage(Module):
    """
    This class is used to compute the derivative of the node
    """
    def __init__(self, delta_x=None):
        super(NodeSpatialAverage, self).__init__()
        self.delta_x = delta_x

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        """
        This function compute the derivative of the node
        It is the mean of the derivative of the edge
        """
        nb_node = x.shape[0]
        average = scatter_mean(edge_attr, edge_index[1], dim=0, dim_size=nb_node)

        return average

class DarcyLoss(Module):
    """
    We rework the dartcy loss function with some better finite differente sheme

    For the flow loss we have the equation :
    -div(a(x) * nabla(u(x))) = f(x)

    The discrete scheme will be to minimise the following loss function :
    -div(a(x) * nabla(u(x))) - f(x) = 0

    We approximate the nabla by the finite difference scheme
    nabla(u(x)) = (u(x + delta_x/2) - u(x)) + (u(x) - u(x - delta_x/2))/delta_x (x component)
                  (u(x + delta_y/2) - u(x)) + (u(x) - u(x - delta_y/2))/delta_y (y component)

    We approximate the div by the finite difference scheme
    div(a(x) * nabla(u(x))) = [a(x + delta_x/2) * (u_(x+delta_x) - u_x)/delta_x - a(x - delta_x/2) * (u_(x) - u_(x-delta_x))/delta_x]/delta_x +
                              [a(x + delta_y/2) * (u_(x+delta_y) - u_x)/delta_y - a(x - delta_y/2) * (u_(x) - u_(x-delta_y))/delta_y]/delta_y
    """
    def __init__(self, delta_x=0.1, delta_y=0.1, index_derivative_node=0, index_derivative_x=0, index_derivative_y=1):
        super(DarcyLoss, self).__init__()
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.index_derivative_node = index_derivative_node
        self.index_derivative_x = index_derivative_x
        self.index_derivative_y = index_derivative_y

        self.edge_spatial_average = EdgeSpatialAverage()
        self.node_spatial_average = NodeSpatialAverage(delta_x=delta_x)

        self.meta_layer = MetaLayer(self.edge_spatial_average, self.node_spatial_average)

        self.spatial_derivator_x = SpatialDerivativeOperator(index_derivative_node, index_derivative_edge=index_derivative_x)
        self.spatial_derivator_y = SpatialDerivativeOperator(index_derivative_node, index_derivative_edge=index_derivative_y)

    def compute_filter_edge_average(self, graph, bool_mask_edge):
        """
        This function compute the filter for the edge
        """
        edge_attrib_tmp = graph.edge_attr[bool_mask_edge, :]
        edge_index_tmp = graph.edge_index[:, bool_mask_edge]

        # we compute the average of the edge
        node_attrib_average, _, _ = self.meta_layer(graph.x, edge_index_tmp, edge_attrib_tmp)

        return node_attrib_average

    def compute_local_derivative(self, graph, bool_mask_edge, index_derivative):
        """
        This function compute the local derivative of the edge
        """
        edge_attrib_tmp = graph.edge_attr[bool_mask_edge, :]
        edge_index_tmp = graph.edge_index[:, bool_mask_edge]

        # we compute the local derivative of the edge
        if index_derivative == 0:
            # create graph
            graph_tmp = Data(x=graph.x, edge_index=edge_index_tmp, edge_attr=edge_attrib_tmp)
            node_attrib_derivative = self.spatial_derivator_x(graph_tmp)
        elif index_derivative == 1:
            graph_tmp = Data(x=graph.x, edge_index=edge_index_tmp, edge_attr=edge_attrib_tmp)
            node_attrib_derivative = self.spatial_derivator_y(graph_tmp)

        return node_attrib_derivative

    def forward(self, out, a_x, f=1., mask=None):
        """
        out and a_x are graphs
        out is the output of the neural network (FNO)
        a_x is the PDE input 
        """
        edge_index = out.edge_index
        edge_attr = out.edge_attr

        # first we compute the a(x+delta_x/2) and a(x-delta_x/2)
        # we filter attribute of the edge to keep only edge which have non-zero delta_x and that are positive
        bool_index = (edge_attr[:, self.index_derivative_x] > 0)
        a_x_plus_delta_x_2 = self.compute_filter_edge_average(a_x, bool_index)

        # we do the same thing for the negative delta_x
        bool_index = (edge_attr[:, self.index_derivative_x] < 0)
        a_x_minus_delta_x_2 = self.compute_filter_edge_average(a_x, bool_index)

        # we compute the a(x+delta_y/2) and a(x-delta_y/2)
        # we filter attribute of the edge to keep only edge which have non-zero delta_y and that are positive
        bool_index = (edge_attr[:, self.index_derivative_y] > 0)
        a_y_plus_delta_y_2 = self.compute_filter_edge_average(a_x, bool_index)

        # we do the same thing for the negative delta_y
        bool_index = (edge_attr[:, self.index_derivative_y] < 0)
        a_y_minus_delta_y_2 = self.compute_filter_edge_average(a_x, bool_index)

        # we compute the derivative of the node for x and y
        # in upward and backward direction
        bool_index = (edge_attr[:, self.index_derivative_x] > 0)
        u_x_plus_delta_x = self.compute_local_derivative(out, bool_index, self.index_derivative_x)

        bool_index = (edge_attr[:, self.index_derivative_x] < 0)
        u_x_minus_delta_x = self.compute_local_derivative(out, bool_index, self.index_derivative_x)

        bool_index = (edge_attr[:, self.index_derivative_y] > 0)
        u_y_plus_delta_y = self.compute_local_derivative(out, bool_index, self.index_derivative_y)

        bool_index = (edge_attr[:, self.index_derivative_y] < 0)
        u_y_minus_delta_y = self.compute_local_derivative(out, bool_index, self.index_derivative_y)

        # now we can compute the loss
        loss = (a_x_plus_delta_x_2 * u_x_plus_delta_x - a_x_minus_delta_x_2 * u_x_minus_delta_x)/self.delta_x + \
                (a_y_plus_delta_y_2 * u_y_plus_delta_y - a_y_minus_delta_y_2 * u_y_minus_delta_y)/self.delta_y + f

        print(a_x_plus_delta_x_2)

        if mask is not None:
            loss = loss[mask]

        return loss