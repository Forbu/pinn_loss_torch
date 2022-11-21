import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator
from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator
from torch_geometric.data import Data

from discretize_pinn_loss.pdebenchmark_darcy import create_darcy_graph

import torch

def test_nabla2d_operator():
    # we create the operator
    nabla2d_operator = Nabla2DOperator(delta_x=1, delta_y=1)

    x = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, 2)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d = nabla2d_operator(graph)

    assert nabla2d.shape == (100, 2)

def test_nabla2d_product_operator():

    # we create the operator
    nabla2d_product_operator = Nabla2DProductOperator(delta_x=1, delta_y=1)

    x = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, 2)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d_product = nabla2d_product_operator(graph)

    assert nabla2d_product.shape == (100, )

def test_darcy_flow_operator():

    # we create the operator
    darcy_flow_operator = DarcyFlowOperator(delta_x=1, delta_y=1)

    x = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, 2)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    a_x = torch.randn(100, 1)
    f = 4

    # create the graph
    graph_a_x = Data(x=a_x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    darcy_flow = darcy_flow_operator(graph, graph_a_x, f)

    assert darcy_flow.shape == (100, )

def test_darcy_flow_operator_2():
    """
    We try to test backward pass on the dartcy flow operator
    """
    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # we create the operator
    darcy_flow_operator = DarcyFlowOperator(delta_x=delta_x, delta_y=delta_y)

    x = torch.randn(nb_nodes, 2, requires_grad=True)/10.
    x = torch.nn.Parameter(x)

    # require grad for x

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    a_x = torch.randn(nb_nodes, 1, requires_grad=True)


    f = 4

    # create the graph
    graph_a_x = Data(x=a_x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    darcy_flow = darcy_flow_operator(graph, graph_a_x, f)

    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(darcy_flow, torch.zeros_like(darcy_flow))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    assert not torch.isnan(x.grad).any()

    assert x.grad is not None

def test_spatial_backward():
    """
    We try to test backward pass on the dartcy flow operator
    combined with FNO operator
    """
    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # we create the operator
    nabla2d_operator = SpatialDerivativeOperator(index_derivative_edge=0, index_derivative_node=0)

    x = torch.randn(nb_nodes, 2, requires_grad=True)

    # we remove edge where edge_attr is 0
    edge_index = edge_index[:, edge_attr[:, 0] != 0]
    edge_attr = edge_attr[edge_attr[:, 0] != 0]

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d = nabla2d_operator(graph)

    print("nabla2d", nabla2d.shape)

    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(nabla2d, torch.zeros_like(nabla2d))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    assert not torch.isnan(x.grad).any()

    assert x.grad is not None


def test_nabla_backward():
    """
    We try to test backward pass on the dartcy flow operator
    combined with FNO operator
    """
    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # we create the operator
    nabla2d_operator = Nabla2DOperator(delta_x, delta_y, index_derivative_node=0,
                                             index_derivative_x=0, index_derivative_y=1)

    x = torch.randn(nb_nodes, 2, requires_grad=True)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d = nabla2d_operator(graph)

    print("nabla2d", nabla2d.shape)

    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(nabla2d, torch.zeros_like(nabla2d))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    assert not torch.isnan(x.grad).any()
    assert x.grad is not None

def test_nabla2dProduct_backward():

    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # we create the operator
    nabla2d_product_operator = Nabla2DProductOperator(delta_x, delta_y, index_derivative_node=0,
                                             index_derivative_x=0, index_derivative_y=1)

    x = torch.randn(nb_nodes, 2, requires_grad=True)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d_product = nabla2d_product_operator(graph)

    print("nabla2d_product", nabla2d_product.shape)

    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(nabla2d_product, torch.zeros_like(nabla2d_product))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    print(x.grad)

    # check if their is no any nan
    assert not torch.isnan(x.grad).any()

    assert x.grad is not None

def test_fulldarcy_backward():
    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # loss and check if their is no any nan
    loss_fn = torch.nn.MSELoss()

    x = torch.randn(nb_nodes, 2, requires_grad=True)
    x = x/10.

    # require grad for x
    x = torch.nn.Parameter(x)
    

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    a_x = torch.randn(nb_nodes, 1, requires_grad=True)

    # init nabla2d operator

    nabla2d_operator = Nabla2DOperator(delta_x, delta_y, index_derivative_node=0,
                                        index_derivative_x=0, index_derivative_y=1)

    # we compute the nabla2d
    nabla2d = nabla2d_operator(graph)

    print("nabla2d", nabla2d)

    # we compute the nabla2d product
    nabla2d_product_operator = Nabla2DProductOperator(delta_x, delta_y, index_derivative_node=0,    
                                                        index_derivative_x=0, index_derivative_y=1) 


    tmp_product_a_x_nabla2d = nabla2d * a_x

    print("tmp_product_a_x_nabla2d", tmp_product_a_x_nabla2d)

    # we create the graph
    graph_tmp_product_a_x_nabla2d = Data(x=tmp_product_a_x_nabla2d, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d product
    nabla2d_product = nabla2d_product_operator(graph_tmp_product_a_x_nabla2d)

    print("nabla2d_product", nabla2d_product)

    # loss and check if their is no any nan
    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(nabla2d_product, torch.zeros_like(nabla2d_product))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    print(x.grad)

    # check if their is no any nan
    assert not torch.isnan(x.grad).any()

    assert x.grad is not None


    