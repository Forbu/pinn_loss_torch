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

"""
def test_derivative_operator():

    # we create the model
    derivative_operator = SpatialDerivativeOperator(index_derivative_edge=0, index_derivative_node=0)

    nb_space = 100

    delta_x = 2.0/nb_space

    delta_t = 0.0005

    # initial condition
    x  = torch.linspace(-1, 1, nb_space)
    u_0 = - torch.sin(torch.pi*x)

    u_0 = u_0.unsqueeze(1)
    u_current = u_0

    # we create the graph
    edges, edges_index, mask = create_graph_burger(nb_space, delta_x)

    edges = torch.Tensor(edges)
    edges_index = torch.Tensor(edges_index).long().T

    # plot the result
    import matplotlib.pyplot as plt

    for i in range(700):

        # we create the graph
        nodes = u_current

        # boundary condition
        nodes[0] = 0
        nodes[-1] = 0

        u_current = nodes

        graph = Data(x=nodes, edge_index=edges_index, edge_attr=edges)

        # we compute the derivative
        with torch.no_grad():
            #u_derivative = derivative_operator(graph).reshape((-1, 1)) 

            #graph_x = Data(x=u_derivative, edge_index=edges_index, edge_attr=edges)

            #u_xx = derivative_operator(graph_x).reshape((-1, 1))

            u_derivative = torch.zeros_like(u_current)
            u_derivative[1:-1, :] = (graph.x[1:-1, :] - graph.x[:-2, :] )/(delta_x)

            u_xx = torch.zeros_like(u_current)
            u_xx[1:-1, :] = (graph.x[2:, :] - 2*graph.x[1:-1, :] + graph.x[:-2, :] )/(delta_x**2)

            #print(u_derivative[1:-1, :])
            #print(u_derivative[1:-1, :])

            u_dissipative = 0#0.0001*u_xx

        # we compute the new value
        u_new = u_current + delta_t * ( - u_derivative * u_current + u_dissipative)
        if i%60 == 0:
            #plt.plot(x, u_derivative.detach().numpy(), label=f"t={i*delta_t}")
            plt.plot(x, u_current.detach().numpy(), label=f"t={i*delta_t}")
        

        u_current = u_new

    plt.legend()

    plt.show()

    # save fig
    plt.savefig("test_derivative_operator.png")


    assert False

    assert out.shape == (nb_space,)
"""




