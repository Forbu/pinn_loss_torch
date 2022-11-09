"""
Here we can test the GNN model
"""

import pytest

from discretize_pinn_loss.models_graph import GNN

import torch

# we create a dummy graph
from torch_geometric.data import Data

def test_gnn():

    in_dim_node = 2
    in_dim_edge = 2
    out_dim = 2

    # we create the model
    model = GNN(
            in_dim_node, #includes data window, node type, inlet velocity 
            in_dim_edge, #distance and relative coordinates
            out_dim, #includes x-velocity, y-velocity, volume fraction, pressure (or a subset)
            out_dim_node=32, out_dim_edge=32, 
            hidden_dim_node=32, hidden_dim_edge=32,
            hidden_layers_node=2, hidden_layers_edge=2,
            #graph processor attributes:
            mp_iterations=5,
            hidden_dim_processor_node=32, hidden_dim_processor_edge=32, 
            hidden_layers_processor_node=2, hidden_layers_processor_edge=2,
            mlp_norm_type='LayerNorm',
            #decoder attributes:
            hidden_dim_decoder=128, hidden_layers_decoder=2,
            output_type='acceleration')

    x = torch.randn(100, in_dim_node)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, in_dim_edge)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    out = model(graph)

    assert out.shape == (100, out_dim)