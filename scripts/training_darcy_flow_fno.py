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


from discretize_pinn_loss.pdebenchmark_darcy import Darcy2DPDEDataset
from discretize_pinn_loss.models_fno import FNO2d
from discretize_pinn_loss.losses_darcy import DarcyFlowOperator

import pytorch_lightning as pl

import torch
from torch_geometric.data import Data, DataLoader, Batch

import hashlib
import datetime
import math

import os

import wandb

import matplotlib.pyplot as plt

import einops

torch.manual_seed(42)

class FnoFull(pl.LightningModule):
    def __init__(self, model_fno, loss_function):
        super().__init__()

        self.model = model_fno
        self.loss_function = loss_function

        self.loss_mse = torch.nn.MSELoss()

    def forward(self, graph):

        # get the input node and reshape it because the model expects a batch
        nodes = graph.x

        batch_size = graph.ptr.shape[0] - 1

        size_image = math.sqrt(int(nodes.shape[0] / batch_size))

        nodes = einops.rearrange(nodes, "(b n) d -> b n d", b=batch_size)

        images = einops.rearrange(nodes, "b (s sb) d -> b s sb d", s=int(size_image), sb=int(size_image))

        result = self.model(images)

        result = einops.rearrange(result, "b s sb n -> b (s sb) n", s=int(size_image), sb=int(size_image))

        result = einops.rearrange(result, "b n d -> (b n) d")

        return result

    def training_step(self, batch, batch_idx):
        
        graph_a_x = batch
        edge_index = graph_a_x.edge_index
        edges = graph_a_x.edge_attr


        # forward pass
        nodes_pred = self.forward(graph_a_x)

        # we create the two graphs
        graph_pred = Data(x=nodes_pred, edge_index=edge_index, edge_attr=edges)

        print("graph_pred.x", graph_pred.x)
        print("graph_a_x.x", graph_a_x.x)

        # compute loss
        relative_loss = self.loss_function(graph_pred, graph_a_x)

        print("relative_loss", relative_loss)

        loss = self.loss_mse(relative_loss, torch.zeros_like(relative_loss))

        self.log("train_loss", loss)

        print(loss)
        exit()

        return loss

    def validation_step(self, batch, batch_idx):

        a_x = batch

        # forward pass
        u_x = self.forward(a_x)

        # we create the two graphs
        graph_pred = Data(x=u_x, edge_index=a_x.edge_index, edge_attr=a_x.edge_attr)

        # compute loss
        relative_loss = self.loss_function(graph_pred, a_x)

        loss = self.loss_mse(relative_loss, torch.zeros_like(relative_loss))

        self.log("val_loss", loss)

        return loss

    # on validation epoch end
    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

def train():

    # create domain
    nb_space = 128

    delta_x = 1.0 / nb_space
    delta_y = delta_x

    batch_size = 8

    # init dataset and dataloader
    path = "/app/data/2D_DarcyFlow_beta1.0_Train.hdf5"
    dataset = Darcy2DPDEDataset(path_hdf5=path, delta_x=delta_x, delta_y=delta_y)

    # divide into train and test
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # init model
    input_dim = 3
    modes = 16
    width = 32

    # we create the model
    model = FNO2d(modes1=modes, modes2=modes,  width=width, input_dim=input_dim)

    # we create the burger function
    burger_loss = DarcyFlowOperator(index_derivative_node=0, index_derivative_x=0, index_derivative_y=1, delta_y=delta_y, delta_x=delta_x)

    # we create the trainer
    gnn_full = FnoFull(model, burger_loss)

    # we create the trainer with the logger
    # init wandb key
    # read wandb_key/key.txt file to get the key
    with open("/app/wandb_key/key.txt", "r") as f:
        key = f.read()

    os.environ['WANDB_API_KEY'] = key
    #wandb.init(project='2D_Darcy', entity='forbu14')
    
    wandb_logger = None #pl.loggers.WandbLogger(project="2D_Darcy", name="2D_Darcy")
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, gradient_clip_val=0.5, accumulate_grad_batches=2)

    # we train
    trainer.fit(gnn_full, dataloader_train, dataloader_test)

    # we save the model
    torch.save(trainer.model.state_dict(), "models_params/darcy_model.pt")

    return gnn_full

if __name__ == "__main__":
    train()

