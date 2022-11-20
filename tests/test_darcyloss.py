import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator
# we create a dummy graph
from torch_geometric.data import Data

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

    nb_nodes = 128*128

    # we create the operator
    darcy_flow_operator = DarcyFlowOperator(delta_x=0.01, delta_y=0.01)

    x = torch.randn(nb_nodes, 2, requires_grad=True)
    edge_index = torch.randint(0, nb_nodes, (2, nb_nodes))
    edge_attr = torch.randn(nb_nodes, 2)

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

    # we compute the gradient
    loss.backward()

    assert x.grad is not None
    assert a_x.grad is not None

def test_darcy_flow_operator_fno():
    """
    We try to test backward pass on the dartcy flow operator
    combined with FNO operator
    """

    pass