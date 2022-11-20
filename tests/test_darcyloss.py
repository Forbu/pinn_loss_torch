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

    a_x = torch.randn(100, 2)
    f = 4

    # we compute the nabla2d
    darcy_flow = darcy_flow_operator(graph, a_x, f)

    assert darcy_flow.shape == (100, )