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

import sys
sys.path.append("/app/")


from discretize_pinn_loss.pdebenchmark import BurgerPDEDataset
from discretize_pinn_loss.models import GNN
from discretize_pinn_loss.utils import create_graph_burger
from discretize_pinn_loss.loss_operator import BurgerDissipativeLossOperator

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

class GnnFull(pl.LightningModule):
    def __init__(self, model_gnn, loss_function):
        super().__init__()

        self.model = model_gnn
        self.loss_function = loss_function

        self.loss_mse = torch.nn.MSELoss()

    def forward(self, graph):
        return self.model(graph)

    def training_step(self, batch, batch_idx):
        
        nodes = batch["nodes"].squeeze(0).unsqueeze(1).float()
        edges = batch["edges"].squeeze(0).float()
        edge_index = batch["edges_index"].squeeze(0).long().T

        graph_t_1 = Data(x=nodes, edge_index=edge_index, edge_attr=edges)

        # forward pass
        nodes_pred = self.forward(graph_t_1)

        # we create the two graphs
        graph_pred = Data(x=nodes_pred, edge_index=edge_index, edge_attr=edges)

        # compute loss
        relative_loss = self.loss_function(graph_pred, graph_t_1)

        loss = self.loss_mse(relative_loss, torch.zeros_like(relative_loss))

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        nodes = batch["nodes"].squeeze(0).unsqueeze(1).float()
        edges = batch["edges"].squeeze(0).float()
        edge_index = batch["edges_index"].squeeze(0).long().T

        graph_t_1 = Data(x=nodes, edge_index=edge_index, edge_attr=edges)

        # forward pass
        nodes_pred = self.forward(graph_t_1)

        # we create the two graphs
        graph_pred = Data(x=nodes_pred, edge_index=edge_index, edge_attr=edges)

        # compute loss
        relative_loss = self.loss_function(graph_pred, graph_t_1)

        loss = self.loss_mse(relative_loss, torch.zeros_like(relative_loss))

        self.log("val_loss", loss)


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

def train():

    # create domain
    nb_space = 1024
    nb_time = 200

    delta_x = 2.0 / nb_space
    delta_t = 2.0 / nb_time # to check

    edges, edges_index = create_graph_burger(nb_space, delta_x, nb_nodes=None, nb_edges=None)

    # init dataset and dataloader
    path = "/app/data/1D_Burgers_Sols_Nu0.01.hdf5"
    dataset = BurgerPDEDataset(path_hdf5=path, edges=edges, edges_index=edges_index)

    # we take a subset of the dataset
    dataset = torch.utils.data.Subset(dataset, range(0, 10000))

    # divide into train and test
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    # init model
    in_dim_node = 1
    in_dim_edge = 1
    out_dim = 1

    # we create the model
    model = GNN(
            in_dim_node, #includes data window, node type, inlet velocity 
            in_dim_edge, #distance and relative coordinates
            out_dim, #includes x-velocity, y-velocity, volume fraction, pressure (or a subset)
            out_dim_node=64, out_dim_edge=64, 
            hidden_dim_node=64, hidden_dim_edge=64,
            hidden_layers_node=2, hidden_layers_edge=2,
            #graph processor attributes:
            mp_iterations=5,
            hidden_dim_processor_node=64, hidden_dim_processor_edge=64, 
            hidden_layers_processor_node=2, hidden_layers_processor_edge=2,
            mlp_norm_type='LayerNorm',
            #decoder attributes:
            hidden_dim_decoder=64, hidden_layers_decoder=2,
            output_type='acceleration')

    # we create the burger function
    burger_loss = BurgerDissipativeLossOperator(index_derivative_node=0, index_derivative_edge=0, delta_t=delta_t, mu=0.01)

    # we create the trainer
    gnn_full = GnnFull(model, burger_loss)

    # we create the trainer
    mlflow_logger = pl.loggers.MLFlowLogger(experiment_name="burger", tracking_uri="http://localhost:5000")
    trainer = pl.Trainer(max_epochs=1, logger=mlflow_logger, gradient_clip_val=0.5, accumulate_grad_batches=8, val_check_interval = 0.05)

    # we train
    trainer.fit(gnn_full, dataloader_train, dataloader_test)

    # we save the model
    torch.save(trainer.model.state_dict(), "models_params/burger_model.pt")

def eval_model_recurssive_mode():
    pass

if __name__ == "__main__":
    train()

