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


from discretize_pinn_loss.pdebenchmark import BurgerPDEDataset, BurgerPDEDatasetFullSimulation
from discretize_pinn_loss.models_fno import FNO1d
from discretize_pinn_loss.utils import create_graph_burger
from discretize_pinn_loss.loss_operator import BurgerDissipativeLossOperator, BurgerDissipativeImplicitLossOperator

import pytorch_lightning as pl

import torch
from torch_geometric.data import Data, DataLoader, Batch

import hashlib
import datetime

import os

import wandb

import matplotlib.pyplot as plt

import einops

torch.manual_seed(42)

class FnoFull(pl.LightningModule):
    def __init__(self, model_fno, loss_function, eval_dataset_full_image=None):
        super().__init__()

        self.model = model_fno
        self.loss_function = loss_function
        self.eval_dataset_full_image = eval_dataset_full_image

        self.loss_mse = torch.nn.MSELoss()
        self.baseline = torch.nn.Identity()

    def forward(self, graph):

        # get the input node and reshape it because the model expects a batch
        nodes = graph.x

        batch_size = graph.ptr.shape[0] - 1

        nodes = einops.rearrange(nodes, "(b n) d -> b n d", b=batch_size)

        result = self.model(nodes)

        # we reshape the result to have the same shape as the input
        result = result.reshape(-1, 1)

        return result

    def training_step(self, batch, batch_idx):
        
        graph_t_1 = batch
        edge_index = graph_t_1.edge_index
        edges = graph_t_1.edge_attr
        nodes = graph_t_1.x
        mask = graph_t_1.mask

        # forward pass
        nodes_pred = self.forward(graph_t_1)

        # we create the two graphs
        graph_pred = Data(x=nodes_pred, edge_index=edge_index, edge_attr=edges)

        graph_baseline = Data(x=self.baseline(nodes), edge_index=edge_index, edge_attr=edges)

        # compute loss
        relative_loss = self.loss_function(graph_pred, graph_t_1, mask)

        relative_loss_baseline = self.loss_function(graph_baseline, graph_t_1)

        loss = self.loss_mse(relative_loss, torch.zeros_like(relative_loss))

        loss_baseline = self.loss_mse(relative_loss_baseline, torch.zeros_like(relative_loss_baseline))

        self.log("train_loss", loss)
        self.log("train_loss_baseline", loss_baseline)

        return loss

    def validation_step(self, batch, batch_idx):

        graph_t_1 = batch

        # forward pass
        nodes_pred = self.forward(graph_t_1)

        # we create the two graphs
        graph_pred = Data(x=nodes_pred, edge_index=graph_t_1.edge_index, edge_attr=graph_t_1.edge_attr)

        # compute loss
        relative_loss = self.loss_function(graph_pred, graph_t_1)

        loss = self.loss_mse(relative_loss, torch.zeros_like(relative_loss))

        self.log("val_loss", loss)

        return loss

    # on validation epoch end
    def validation_epoch_end(self, outputs):

        if self.eval_dataset_full_image is None:
            return

        # here we should retrieve the full image from the dataset and compute the result with the help of the model (recursively)
        # we should then compute the loss between the prediction and the ground truth and log the artifact
        # we should also log the loss
        ############## Checking result with a random sample ####################
        # we choose a random sample
        id_sample = torch.randint(0, len(self.eval_dataset_full_image), (1,)).item()
        data_full = self.eval_dataset_full_image[id_sample]

        # get boundary conditions
        nodes_t0 = data_full["nodes_t0"].float().to(self.device)
        edges = data_full["edges"].float().to(self.device)
        edges_index = data_full["edges_index"].long().T.to(self.device)
        image_result = data_full["image_result"].float().to(self.device)

        mask = data_full["mask"].float().to(self.device)

        nodes_boundary_x__1 = data_full['nodes_boundary_x__1'].to(self.device)
        nodes_boundary_x_1 = data_full['nodes_boundary_x_1'].to(self.device)

        self.recursive_simulation(nodes_t0, edges, edges_index, image_result, mask, nodes_boundary_x__1, nodes_boundary_x_1, name_simulation="simulation_true_value")

        ############## Checking result with a simple sample ####################
        # we generate a simple sample
        #nb_space = nodes_t0.shape[0]
        #nodes_t0 = torch.rand((nb_space, 1)) - 0.5
        #nodes_t0 = torch.cat([nodes_t0, mask], dim=1)

        #self.recursive_simulation(nodes_t0, edges, edges_index, image_result, mask, nodes_boundary_x__1, nodes_boundary_x_1, name_simulation="simulation_random")

        ############## Checking the coherence of the provided solution to the burger loss equation ####################
        self.check_burger_loss_equation(image_result, mask, edges_index, edges)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def recursive_simulation(self, nodes_t0, edges, edges_index, image_result, mask, nodes_boundary_x__1, nodes_boundary_x_1, name_simulation="simulation"):

        result_prediction = torch.zeros_like(image_result)
        result_prediction[0, :] = nodes_t0[:, 0]

        burger_loss_tot = torch.zeros_like(image_result)

        # we create the graph
        graph_current = Data(x=nodes_t0, edge_index=edges_index, edge_attr=edges)

        # create Batch
        graph_current = Batch.from_data_list([graph_current])

        # now we can apply the model recursively to get the full image
        nb_time = image_result.shape[0]

        # forward pass
        for t in range(1, nb_time-1):
            with torch.no_grad():
                nodes_pred = self.forward(graph_current)


            # we add the boundary conditions
            limit_x__1 = nodes_boundary_x_1[t+1]
            limit_x_1 = nodes_boundary_x__1[t+1]

            limit_condition = torch.zeros((nodes_pred.shape[0], 1)).to(self.device)
            limit_condition[0, 0] = limit_x__1
            limit_condition[-1, 0] = limit_x_1
            
            nodes_pred = torch.cat([nodes_pred, mask, limit_condition], axis = 1)

            # we create the two graphs
            graph_pred = Data(x=nodes_pred, edge_index=edges_index, edge_attr=edges)
            graph_pred = Batch.from_data_list([graph_pred])

            # compute the burger loss
            with torch.no_grad():
                relative_loss = self.loss_function(graph_pred, graph_current)

            # we update the graph
            graph_current = graph_pred

            # we update the result
            result_prediction[t, :] = nodes_pred[:, 0]

            # we update the burger loss
            burger_loss_tot[t, :] = relative_loss

        # we compute the loss
        loss = self.loss_mse(result_prediction, image_result)

        self.log("val_loss_full_image", loss)

        loss_burger_tot = self.loss_mse(burger_loss_tot, torch.zeros_like(burger_loss_tot))

        self.log("val_loss_burger_pred", loss_burger_tot)

        # we log the two images (ground truth and prediction)
        image_target = image_result.detach().cpu().numpy()
        image_prediction = result_prediction.detach().cpu().numpy()
        
        # get min max 
        min_val = min(image_target.min(), image_prediction.min())
        max_val = max(image_target.max(), image_prediction.max())


        image_name_target = f"images/{name_simulation}_target.png"
        image_name_prediction = f"images/{name_simulation}_prediction.png"

        # we generate the png files using matplotlib
        plt.figure(figsize=(20, 20))
        plt.imshow(image_target, vmin=min_val, vmax=max_val)
        plt.colorbar()
        plt.savefig(image_name_target)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.imshow(image_prediction, vmin=min_val, vmax=max_val)
        plt.colorbar()
        plt.savefig(image_name_prediction)
        plt.close()

        # get run id
        try:
            # call the logger to log the artifact
            self.logger.log_image(key = "train_" + name_simulation, images = [image_name_target], step = self.current_epoch)
            self.logger.log_image(key = "prediction_" + name_simulation, images = [image_name_prediction], step = self.current_epoch)
        except Exception as e:
            print(e)

    def check_burger_loss_equation(self, image_result, mask=None, edges_index=None, edges_attr=None):

        # we compute the burger loss equation for the corresponding image by just computing the burger loss for each time step

        nb_space = image_result.shape[1]
        nb_time = image_result.shape[0]

        # we compute the loss for each time step
        loss_burger = torch.zeros((nb_time, nb_space))

        for t in range(1, nb_time):
            # we compute the loss for the current time step
            # 1. define graph for t and t-1
            nodes_t0 = image_result[t-1, :].reshape(-1, 1)
            # cat with mask
            nodes_t0 = torch.cat([nodes_t0, mask], axis = 1)

            nodes_t1 = image_result[t, :].reshape(-1, 1)
            # cat with mask
            nodes_t1 = torch.cat([nodes_t1, mask], axis = 1)

            # 2. transform into graph
            graph_t0 = Data(x=nodes_t0, edge_index=edges_index, edge_attr=edges_attr)
            graph_t1 = Data(x=nodes_t1, edge_index=edges_index, edge_attr=edges_attr)

            # 3. compute the loss
            loss_burger[t, :] = self.loss_function(graph_t1, graph_t0, mask)

        # we compute the mean loss
        loss_burger_full = self.loss_mse(loss_burger, torch.zeros_like(loss_burger))

        self.log("val_loss_burger_perfect_simulation", loss_burger_full)

def train():

    # create domain
    nb_space = 1024
    nb_time = 200

    delta_x = 2.0 / nb_space
    delta_t = 2.0 / nb_time # to check

    batch_size = 16

    edges, edges_index, mask = create_graph_burger(nb_space, delta_x, nb_nodes=None, nb_edges=None)

    # init dataset and dataloader
    path = "/app/data/1D_Burgers_Sols_Nu0.01.hdf5"
    dataset = BurgerPDEDataset(path_hdf5=path, edges=edges, edges_index=edges_index, mask=mask)

    # init dataset for full image
    dataset_full_image = BurgerPDEDatasetFullSimulation(path_hdf5=path, edges=edges, edges_index=edges_index, mask=mask)

    # we take a subset of the dataset
    dataset = torch.utils.data.Subset(dataset, range(0, 80000))

    # divide into train and test
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # init model
    input_dim = 3
    in_dim_edge = 1
    out_dim = 1

    modes = 16
    width = 64

    # we create the model
    model = FNO1d(modes, width, delta_t=delta_t, input_dim=input_dim)

    # we create the burger function
    burger_loss = BurgerDissipativeImplicitLossOperator(index_derivative_node=0, index_derivative_edge=0, delta_t=delta_t, mu=0.01)

    # we create the trainer
    gnn_full = FnoFull(model, burger_loss, eval_dataset_full_image=dataset_full_image)

    # we create the trainer with the logger
    # init wandb key
    # read wandb_key/key.txt file to get the key
    with open("/app/wandb_key/key.txt", "r") as f:
        key = f.read()

    os.environ['WANDB_API_KEY'] = key
    wandb.init(project='1D_Burgers', entity='forbu14')
    
    wandb_logger = pl.loggers.WandbLogger(project="1D_Burgers", name="GNN_1D_Burgers")
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, gradient_clip_val=0.5, accumulate_grad_batches=2)

    # we train
    trainer.fit(gnn_full, dataloader_train, dataloader_test)

    # we save the model
    torch.save(trainer.model.state_dict(), "models_params/burger_model.pt")

if __name__ == "__main__":
    train()

