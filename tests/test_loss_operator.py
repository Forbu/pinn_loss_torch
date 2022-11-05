"""
tests for loss operator
"""

import pytest

from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator, TemporalDerivativeOperator, BurgerDissipativeLossOperator

from discretize_pinn_loss.utils import create_graph_burger

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

def test_derivative_operator():

    # we create the model
    derivative_operator = SpatialDerivativeOperator(index_derivative_edge=0, index_derivative_node=0)

    nb_space = 1024

    delta_x = 2.0/nb_space

    # initial condition
    x  = torch.linspace(-1, 1, nb_space)
    u_0 = torch.sin(torch.pi*x)

    # we create the graph
    edges, edges_index, mask = create_graph_burger(nb_space, delta_x)

    # we create the graph
    graph = Data(x=u_0, edge_index=edges_index, edge_attr=edges)

    # pass the graph to the model
    out = derivative_operator(graph)

    assert out.shape == (nb_space,)





