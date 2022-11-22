import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator
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














