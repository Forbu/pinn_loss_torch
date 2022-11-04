# training / eval for burger equation

"""
This is the main file to train the GNN on the burger equation

The burger equation is a simple PDE that can be solved with a GNN (and a pinn loss).
The PDE is :
    u_t + u*u_x = mu * u_xx

With the following boundary conditions:
    u(-1, t) = 0
    u(1, t) = 0
    u(x, 0) = sin(pi * x)

The goal is to train a GNN to solve this PDE.
"""

from discretize_pinn_loss.pdebenchmark import BurgerPDEDataset
from discretize_pinn_loss.models import GNN
from discretize_pinn_loss.utils import create_graph_burger

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader

class GnnTrainer(pl.LightningModule):
    def __init__(self, model_gnn, loss_function):
        super().__init__()

        self.model = model_gnn
        self.loss_function = loss_function

        self.loss = torch.nn.MSELoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        loss = self.model.loss(batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

def train():

    # create domain
    nb_space = 1024
    nb_time = 200

    delta_x = 2.0 / nb_space
    delta_t = 2.0 / nb_time # to check

    edges, edges_index = create_graph_burger(nb_space, delta_x, nb_nodes=None, nb_edges=None)

    # init dataset and dataloader
    path = "/app/data/1D_Burgers_Sols_Nu0.001.hdf5"
    dataset = BurgerPDEDataset(path_hdf5=path, edges=edges, edges_index=edges_index)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # init model
    in_dim_node = 1
    in_dim_edge = 1
    out_dim = 1

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
            hidden_dim_decoder=32, hidden_layers_decoder=2,
            output_type='acceleration')

    #

