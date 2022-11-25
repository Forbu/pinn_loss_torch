import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator, DarcyLoss
from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator
from discretize_pinn_loss.pdebenchmark_darcy import create_darcy_graph
from discretize_pinn_loss.pdebenchmark_darcy import Darcy2DPDEDataset, create_darcy_graph

from torch_geometric.data import Data, DataLoader, Batch

from torch_geometric.data import Data

import torch

import einops

import matplotlib.pyplot as plt

def test_darcyflow_operator():
    # we create the operator
    delta_x = 1./128
    delta_y = 1./128

    index_derivative_node = 0
    index_derivative_x = 0
    index_derivative_y = 1

    img_size = 128

    # load dataset
    # init dataset and dataloader
    path = "/app/data/2D_DarcyFlow_beta1.0_Train.hdf5"
    dataset = Darcy2DPDEDataset(path_hdf5=path, delta_x=delta_x, delta_y=delta_y)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # extract one batch
    batch = next(iter(dataloader))

    nabla2d_operator = Nabla2DOperator(delta_x=delta_x, delta_y=delta_y, index_derivative_node=index_derivative_node,
                                            index_derivative_x=index_derivative_x, index_derivative_y=index_derivative_y)
    nabla2d_product_operator = Nabla2DProductOperator(delta_x=delta_x, delta_y=delta_y,
                                            index_derivative_node=index_derivative_node, index_derivative_x=index_derivative_x, index_derivative_y=index_derivative_y)


    a_x = batch
    target = batch.target
    mask = batch.mask

    # we compute the nabla2d
    out = Data(x=target, edge_index=batch.edge_index, edge_attr=batch.edge_attr)

    out_reshape = einops.rearrange(out.x, '(h w) c -> h w c', h=img_size, w=img_size)

    estimated_nabla = torch.zeros((out_reshape.shape[0], out_reshape.shape[1], 2))

    print(out_reshape.shape)
    estimated_nabla[1:-1, :, 0] = (out_reshape[2:, :, 0] - out_reshape[:-2, :, 0]) / (2 * delta_x)
    estimated_nabla[:, 1:-1, 1] = (out_reshape[:, 2:, 0] - out_reshape[:, :-2, 0]) / (2 * delta_y)

    nabla2d = nabla2d_operator(out)


    # product between nabla2d and a_x
    tmp_nabla = nabla2d * a_x.x[:, [index_derivative_node]]

    print(tmp_nabla.shape)

    # we create plot for the two resulting images
    # 1. reshape from graph to image (n, 2) -> (128, 128, 2)
    tmp_nabla_flatten = einops.rearrange(tmp_nabla, '(h w) c -> h w c', h=img_size, w=img_size)

    print(a_x.x[:, [index_derivative_node]].shape)

    a_x_flatten = einops.rearrange(a_x.x[:, [index_derivative_node]], '(h w) c -> h w c', h=img_size, w=img_size)

    estimated_nabla = estimated_nabla * a_x_flatten


    # 2. plot
    fig, axs = plt.subplots(1, 6, figsize=(15, 5))
    
    pos0 = axs[0].imshow(tmp_nabla_flatten[:, :, 0], vmin=-1, vmax=1)
    pos1 = axs[1].imshow(tmp_nabla_flatten[:, :, 1], vmin=-1, vmax=1)

    pos2 = axs[2].imshow(estimated_nabla[:, :, 0], vmin=-1, vmax=1)
    pos3 = axs[3].imshow(estimated_nabla[:, :, 1], vmin=-1, vmax=1)

    pos4 = axs[4].imshow(a_x_flatten[:, :, 0], vmin=-1, vmax=1)

    pos5 = axs[5].imshow(out_reshape, vmin=0, vmax=0.6)

    fig.colorbar(pos0, ax=axs[0])
    fig.colorbar(pos1, ax=axs[1])
    fig.colorbar(pos2, ax=axs[2])
    fig.colorbar(pos3, ax=axs[3])
    fig.colorbar(pos4, ax=axs[4])
    fig.colorbar(pos5, ax=axs[5])

    # adding titles
    axs[0].set_title("nabla2d * a_x")
    axs[1].set_title("nabla2d * a_y")
    axs[2].set_title("estimated nabla2d * a_x")
    axs[3].set_title("estimated nabla2d * a_y")
    axs[4].set_title("a_x")
    axs[5].set_title("target")


    # 3. save
    fig.savefig("tmp_nabla.png")

    # create graph from tmp_nabla
    tmp_nabla_graph = Data(x=tmp_nabla, edge_index=batch.edge_index, edge_attr=batch.edge_attr)

    nabla2d_product = nabla2d_product_operator(tmp_nabla_graph)

    print(nabla2d_product.shape)

    # estimated_nabla_product = derivative_x + derivative_y
    estimated_nabla_product = torch.zeros((estimated_nabla.shape[0], estimated_nabla.shape[1], 1))
    estimated_nabla_product[1:-1, 1:-1, 0] = (estimated_nabla[2:, 1:-1, 0] - estimated_nabla[:-2, 1:-1, 0])/(2 * delta_x) +\
                         (estimated_nabla[1:-1, 2:, 1] - estimated_nabla[1:-1, :-2, 1])/(2 * delta_y)
    
    nabla2d_product_flatten = einops.rearrange(nabla2d_product, '(h w) -> h w', h=img_size, w=img_size)

    # plot distribution of values of nabla2d_product
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # use hist and limit the range between -3 and 3
    axs[0].hist(nabla2d_product_flatten.flatten(), bins=100, range=(-3, 3))

    # save in png
    fig.savefig("nabla2d_product_hist.png")
    

    # 2. plot again
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    pos0 = axs[0].imshow(nabla2d_product_flatten[:, :], vmin=-5, vmax=5)
    pos1 = axs[1].imshow(estimated_nabla_product[:, :, 0], vmin=-5, vmax=5)

    fig.colorbar(pos0, ax=axs[0])
    fig.colorbar(pos1, ax=axs[1])

    # 3. save
    fig.savefig("tmp_nabla_product.png")
    
    pde_loss = torch.abs(nabla2d_product + 1.0)

def test_darcyloss():
    """
    In this script we will test the darcy loss (DarcyLoss) and compare it to the old one (DarcyFlowOperator)
    """
    # we create the operator
    delta_x = 1./128
    delta_y = 1./128

    index_derivative_node = 0
    index_derivative_x = 0
    index_derivative_y = 1

    img_size = 128

    # init mse loss function
    mse_loss = torch.nn.MSELoss()

    # load dataset
    # init dataset and dataloader
    path = "/app/data/2D_DarcyFlow_beta1.0_Train.hdf5"
    dataset = Darcy2DPDEDataset(path_hdf5=path, delta_x=delta_x, delta_y=delta_y)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # extract one batch
    batch = next(iter(dataloader))

    # now that we have a batch of data
    # init darcy flow operator
    darcy_flow_operator = DarcyFlowOperator(delta_x=delta_x, delta_y=delta_y)
    darcy_loss_new_operator = DarcyLoss(delta_x=delta_x, delta_y=delta_y)

    with torch.autograd.detect_anomaly():
        # retrieve out and a_x
        out = batch.target

        a_x = batch.x[:, [0]]

        print(a_x.shape)
        print(out.shape)

        # create graph from a_x and out
        a_x_graph = Data(x=a_x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
        out_graph = Data(x=out, edge_index=batch.edge_index, edge_attr=batch.edge_attr)

        #now we can apply the two operators and compute the loss
        darcy_loss = darcy_flow_operator(out_graph, a_x_graph, mask=batch.mask).unsqueeze(1)
        darcy_loss_new, u_x_plus_delta_x, u_x_minus_delta_x, u_y_plus_delta_y, u_y_minus_delta_y, a_x_plus_delta_x_2, a_x_minus_delta_x_2, a_y_plus_delta_y_2, a_y_minus_delta_y_2 = darcy_loss_new_operator(out_graph, a_x_graph, mask=batch.mask)

        
        print(darcy_loss.shape)
        print(darcy_loss_new.shape)

        # compute the loss for each operator
        loss = mse_loss(darcy_loss, torch.zeros_like(darcy_loss))
        loss_new = mse_loss(darcy_loss_new, torch.zeros_like(darcy_loss_new))

        def reshape_tensor(tensor):
            return einops.rearrange(tensor, '(h w) d -> h w d', h=img_size, w=img_size)

        # reshape tensors
        u_x_plus_delta_x = reshape_tensor(u_x_plus_delta_x)
        u_x_minus_delta_x = reshape_tensor(u_x_minus_delta_x)
        u_y_plus_delta_y = reshape_tensor(u_y_plus_delta_y)
        u_y_minus_delta_y = reshape_tensor(u_y_minus_delta_y)
        a_x_plus_delta_x_2 = reshape_tensor(a_x_plus_delta_x_2)
        a_x_minus_delta_x_2 = reshape_tensor(a_x_minus_delta_x_2)
        a_y_plus_delta_y_2 = reshape_tensor(a_y_plus_delta_y_2)
        a_y_minus_delta_y_2 = reshape_tensor(a_y_minus_delta_y_2)

        # reshape loss
        darcy_loss = reshape_tensor(darcy_loss)
        darcy_loss_new = reshape_tensor(darcy_loss_new)


        print("loss: ", loss)
        print("loss_new: ", loss_new)

        # now we can plot the results
        # 1. plot every variable
        fig, axs = plt.subplots(1, 8, figsize=(15, 5))

        pos0 = axs[0].imshow(u_x_plus_delta_x[:, :, 0].reshape(img_size, img_size), vmin=-1, vmax=1)
        pos1 = axs[1].imshow(u_x_minus_delta_x[:, :, 0].reshape(img_size, img_size), vmin=-1, vmax=1)
        pos2 = axs[2].imshow(u_y_plus_delta_y[:, :, 0].reshape(img_size, img_size), vmin=-1, vmax=1)
        pos3 = axs[3].imshow(u_y_minus_delta_y[:, :, 0].reshape(img_size, img_size), vmin=-1, vmax=1)
        pos4 = axs[4].imshow(a_x_plus_delta_x_2[:, :, 0].reshape(img_size, img_size), vmin=-1, vmax=1)
        pos5 = axs[5].imshow(a_x_minus_delta_x_2[:, :, 0].reshape(img_size, img_size), vmin=-1, vmax=1)
        pos6 = axs[6].imshow(a_y_plus_delta_y_2[:, :, 0].reshape(img_size, img_size), vmin=-1, vmax=1)
        pos7 = axs[7].imshow(a_y_minus_delta_y_2[:, :, 0].reshape(img_size, img_size), vmin=-1, vmax=1)

        # titles
        axs[0].set_title("u_x_plus_delta_x")
        axs[1].set_title("u_x_minus_delta_x")
        axs[2].set_title("u_y_plus_delta_y")
        axs[3].set_title("u_y_minus_delta_y")
        axs[4].set_title("a_x_plus_delta_x_2")
        axs[5].set_title("a_x_minus_delta_x_2")
        axs[6].set_title("a_y_plus_delta_y_2")
        axs[7].set_title("a_y_minus_delta_y_2")


        fig.colorbar(pos0, ax=axs[0])
        fig.colorbar(pos1, ax=axs[1])
        fig.colorbar(pos2, ax=axs[2])
        fig.colorbar(pos3, ax=axs[3])
        fig.colorbar(pos4, ax=axs[4])
        fig.colorbar(pos5, ax=axs[5])
        fig.colorbar(pos6, ax=axs[6])
        fig.colorbar(pos7, ax=axs[7])

        # save
        fig.savefig("tmp_darcy_loss_u.png")

        # 2. plot again
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        pos0 = axs[0].imshow(darcy_loss[:, :, 0].reshape(img_size, img_size), vmin=-5, vmax=5)
        pos1 = axs[1].imshow(darcy_loss_new[:, :, 0].reshape(img_size, img_size), vmin=-5, vmax=5)

        fig.colorbar(pos0, ax=axs[0])
        fig.colorbar(pos1, ax=axs[1])

        # 3. save

        fig.savefig("tmp_darcy_loss.png")


    assert False
    













