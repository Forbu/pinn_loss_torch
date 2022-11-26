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


from discretize_pinn_loss.pdebenchmark_darcy import Darcy2DPDEDatasetTemporal
from discretize_pinn_loss.models_fno import FNO2d
from discretize_pinn_loss.losses_darcy import DarcyFlowOperator, DarcyLoss

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

import numpy as np

torch.manual_seed(42)

class FnoFull(pl.LightningModule):
    def __init__(self, model_fno, loss_function, nb_step=10, delta_t=0.01, batch_size_init=8):
        super().__init__()

        self.model = model_fno
        self.loss_function = loss_function

        self.nb_step = nb_step
        self.delta_t = delta_t 

        self.loss_mse = torch.nn.MSELoss()

        self.batch_size_init = batch_size_init

    def forward(self, graph):

        # get the input node and reshape it because the model expects a batch
        nodes = graph.x

        batch_size = self.batch_size_init

        size_image = math.sqrt(int(nodes.shape[0] / batch_size))

        nodes = einops.rearrange(nodes, "(b n) d -> b n d", b=batch_size)
        images = einops.rearrange(nodes, "b (s sb) d -> b s sb d", s=int(size_image), sb=int(size_image))

        result = self.model(images, temporal=True)

        result = einops.rearrange(result, "b s sb n -> b (s sb) n", s=int(size_image), sb=int(size_image))
        result = einops.rearrange(result, "b n d -> (b n) d")

        return result

    def training_step(self, batch, batch_idx):
        
        edge_index = batch.edge_index
        edges = batch.edge_attr

        a_x = batch.a_x
        graph_a_x = Data(x=a_x, edge_index=edge_index, edge_attr=edges)

        graph_current = batch

        loss_total = 0

        for _ in range(self.nb_step):

            # forward pass
            nodes_pred = self.forward(graph_current)

            # we create the two graphs
            graph_pred = Data(x=nodes_pred, edge_index=edge_index, edge_attr=edges)

            # compute loss
            darcy_loss = self.loss_function(graph_pred, graph_a_x).unsqueeze(1)
            temporal_loss = - (nodes_pred - graph_current.x[:, [0]])/self.delta_t

            relative_loss = darcy_loss + temporal_loss 
            loss = self.loss_mse(relative_loss, torch.zeros_like(relative_loss))

            # concat nodes_pred with mask and limit_condition
            nodes_pred = torch.cat([nodes_pred, batch.mask.unsqueeze(1), batch.limit_condition, graph_current.x[:, [3]]], dim=1)

            # create new current graph
            graph_current = Data(x=nodes_pred, edge_index=edge_index, edge_attr=edges)

            loss_total += loss

        # normalize loss
        loss_total /= self.nb_step

        self.log("train_loss", loss_total)

        return loss_total

    def validation_step(self, batch, batch_idx):

        edge_index = batch.edge_index
        edges = batch.edge_attr

        a_x = batch.a_x
        graph_a_x = Data(x=a_x, edge_index=edge_index, edge_attr=edges)

        graph_current = batch

        loss_total = 0

        for _ in range(self.nb_step):

            # forward pass
            nodes_pred = self.forward(graph_current)


            # we create the two graphs
            graph_pred = Data(x=nodes_pred, edge_index=edge_index, edge_attr=edges)

            # compute loss
            darcy_loss = self.loss_function(graph_pred, graph_a_x)
            temporal_loss = - (nodes_pred - graph_current.x[:, [0]])/self.delta_t

            relative_loss = darcy_loss + temporal_loss 

            loss = self.loss_mse(relative_loss, torch.zeros_like(relative_loss))

            # concat nodes_pred with mask and limit_condition
            nodes_pred = torch.cat([nodes_pred, batch.mask.unsqueeze(1), batch.limit_condition, graph_current.x[:, [3]]], dim=1)

            # create new current graph
            graph_current = Data(x=nodes_pred, edge_index=edge_index, edge_attr=edges)

            loss_total += loss
        
        # normalize loss
        loss_total /= self.nb_step

        self.log("val_loss", loss_total)

        # now we can save the result image to wandb for visualization
        # we need to reshape the image to be able to plot it
        batch_size = batch.ptr.shape[0] - 1

        # we should also log the loss with the generated target value coming form the solver (data)
        graph_target= Data(x=batch.target, edge_index=batch.edge_index, edge_attr=batch.edge_attr)

        # create graph a_x
        a_x = batch.a_x
        graph_a_x = Data(x=a_x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)

        relative_loss_solver = torch.abs(self.loss_function(graph_target, graph_a_x, mask=batch.mask.unsqueeze(1)))

        loss_solver = self.loss_mse(relative_loss_solver, torch.zeros_like(relative_loss_solver))

        self.log("val_loss_solver", loss_solver)

        size_image = math.sqrt(int(graph_a_x.x.shape[0] / batch_size))

        u_x = nodes_pred[:, [0]]

        a_x = einops.rearrange(graph_a_x.x, "(b n) d -> b n d", b=batch_size)
        a_x = einops.rearrange(a_x, "b (s sb) d -> b s sb d", s=int(size_image), sb=int(size_image))
        u_x = einops.rearrange(u_x, "(b n) d -> b n d", b=batch_size)
        u_x = einops.rearrange(u_x, "b (s sb) d -> b s sb d", s=int(size_image), sb=int(size_image))

        # same operation for the target
        target = batch.target

        target = einops.rearrange(target, "(b n) d -> b n d", b=batch_size)
        target = einops.rearrange(target, "b (s sb) d -> b s sb d", s=int(size_image), sb=int(size_image))

        #same thing on relative loss
        relative_loss = einops.rearrange(relative_loss.squeeze(), "(b n) -> b n", b=batch_size)
        relative_loss = einops.rearrange(relative_loss, "b (s sb) -> b s sb", s=int(size_image), sb=int(size_image))

        #same thing on relative loss solver
        relative_loss_solver = einops.rearrange(relative_loss_solver.squeeze(), "(b n) -> b n", b=batch_size)
        relative_loss_solver = einops.rearrange(relative_loss_solver, "b (s sb) -> b s sb", s=int(size_image), sb=int(size_image))

        # now we can save the result image to wandb for visualization
        # we need to reshape the image to be able to plot it
        # save the image into png format
        fig, axs = plt.subplots(1, 5, figsize=(15, 5))

        pos0 = axs[0].imshow(a_x[0, :, :, 0].detach().cpu().numpy())
        axs[0].set_title("a_x")

        pos1 = axs[1].imshow(u_x[0, :, :, 0].detach().cpu().numpy(), vmin=0, vmax=0.30)
        axs[1].set_title("u_x")

        pos2 = axs[2].imshow(target[0, :, :, 0].detach().cpu().numpy(), vmin=0, vmax=0.30)
        axs[2].set_title("target")

        pos3 = axs[3].imshow(relative_loss[0, :, :].detach().cpu().numpy(), vmin=0, vmax=4)
        axs[3].set_title("relative loss pred")

        pos4 = axs[4].imshow(relative_loss_solver[0, :, :].detach().cpu().numpy(), vmin=0, vmax=4)
        axs[4].set_title("relative loss solver")

        fig.colorbar(pos0, ax=axs[0])
        fig.colorbar(pos1, ax=axs[1])
        fig.colorbar(pos2, ax=axs[2])
        fig.colorbar(pos3, ax=axs[3])
        fig.colorbar(pos4, ax=axs[4])

        # save into png
        fig.savefig("tmp.png")
    
        self.logger.log_image("a_x u_x target", ["tmp.png"])

        return loss

    # on validation epoch end
    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def test_train_step(self, dataloader):

        # init optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        with torch.autograd.detect_anomaly():
            
            for batch in dataloader:
                loss = self.training_step(batch, 0)
                print("loss", loss)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

def train():

    # create domain
    nb_space = 128

    delta_x = 1.0 / nb_space
    delta_y = delta_x

    delta_t = 0.01

    batch_size = 1

    path_init = "/app/discretize_pinn_loss/data_init/init_solution.pt"

    # init dataset and dataloader
    path = "/app/data/2D_DarcyFlow_beta1.0_Train.hdf5"
    dataset = Darcy2DPDEDatasetTemporal(path_hdf5=path, delta_x=delta_x, delta_y=delta_y, path_init=path_init)

    # divide into train and test
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # init model
    input_dim = 4
    modes = 32
    width = 32

    

    # we create the model
    model = FNO2d(modes1=modes, modes2=modes,  width=width, input_dim=input_dim, path_init=path_init)

    # we create the burger function
    darcy_loss = DarcyLoss(index_derivative_node=0, index_derivative_x=0, index_derivative_y=1, delta_y=delta_y, delta_x=delta_x)

    # we create the trainer
    gnn_full = FnoFull(model, darcy_loss, nb_step=5, delta_t=delta_t, batch_size_init=batch_size)

    # we create the trainer with the logger
    # init wandb key
    # read wandb_key/key.txt file to get the key
    with open("/app/wandb_key/key.txt", "r") as f:
        key = f.read()

    os.environ['WANDB_API_KEY'] = key
    wandb.init(project='2D_Darcy', entity='forbu14')
    
    wandb_logger = pl.loggers.WandbLogger(project="2D_Darcy", name="2D_Darcy")
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, gradient_clip_val=0.5, accumulate_grad_batches=2)

    
    # test with test_train_step
    # gnn_full.test_train_step(dataloader_train)

    # we train
    trainer.fit(gnn_full, dataloader_train, dataloader_test)

    # we save the model
    torch.save(trainer.model.state_dict(), "models_params/darcy_model.pt")

    return gnn_full

if __name__ == "__main__":
    train()

