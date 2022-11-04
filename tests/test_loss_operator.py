"""
tests for loss operator
"""

import pytest

from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator, TemporalDerivativeOperator, BurgerDissipativeLossOperator

import torch
from torch_geometric.data import Data

@pytest.fixture
def graph():
    x = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, 2)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph

@pytest.fixture
def graph_t_1():
    x = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, 2)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph
    
def test_spatial_derivative_operator(graph):

    # we create the model
    model = SpatialDerivativeOperator(index_derivative_edge=0, index_derivative_node=0)

    out = model(graph)

    assert out.shape == (100,)

def test_temporal_derivative_operator(graph, graph_t_1):
    
    # we create the model
    model = TemporalDerivativeOperator(index_derivator=0, delta_t=0.01)

    out = model(graph, graph_t_1)

    assert out.shape == (100,)

def test_burger_dissipative_loss_operator(graph, graph_t_1):

    # we create the model
    model = BurgerDissipativeLossOperator(index_derivative_node=0, delta_t=0.01, index_derivative_edge=0, mu=0.1)

    out = model(graph, graph_t_1)

    assert out.shape == (100,)




