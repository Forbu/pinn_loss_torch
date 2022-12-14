import pytest 

from discretize_pinn_loss.pdebenchmark_burger import BurgerPDEDataset, BurgerPDEDatasetFullSimulation, BurgerPDEDatasetMultiTemporal
from discretize_pinn_loss.utils import create_graph_burger

# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as DataLoader2

import h5py

from discretize_pinn_loss.pdebenchmark_darcy import Darcy2DPDEDataset, create_darcy_graph

import torch

def test_burgerpdedataset():

    # we choose the discretization of the space and the time
    nb_space = 1024
    nb_time = 100

    delta_x = 1.0 / nb_space
    delta_t = 1.0 / nb_time

    # we choose the batch size
    batch_size = 16

    # now we can create the dataloader
    edges, edges_index, mask = create_graph_burger(nb_space, delta_x, nb_nodes=None, nb_edges=None)

    print("edges.shape", edges.shape)
    print("edges_index.shape", edges_index.shape)

    path_hdf5 = "/app/data/1D_Burgers_Sols_Nu0.01.hdf5"

    # we create the dataset
    dataset = BurgerPDEDataset(path_hdf5, edges, edges_index, mask=mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, follow_batch=['mask'])

    # we test the dataloader
    for i, data in enumerate(dataloader):

        print(data)
        
        assert data["x"].shape == (batch_size*nb_space, 3)
        assert data["edge_attr"].shape == (batch_size*(nb_space - 1) * 2 , 1)
        assert data["edge_index"].shape == (2, batch_size*(nb_space - 1) * 2)

        break 


def test_burgerpdedatasettemporal():

    path_hdf5 = "/app/data/1D_Burgers_Sols_Nu0.01.hdf5"

    nb_space = 1024
    nb_time = 201

    delta_x = 2.0 / nb_space
    delta_t = 1.0 / nb_time

    batch_size = 3

    edges, edges_index, mask = create_graph_burger(nb_space, delta_x, nb_nodes=None, nb_edges=None)

    dataset_temporal = BurgerPDEDatasetMultiTemporal(path_hdf5, mask=None, timesteps=3, edges=edges, edges_index=edges_index)

    dataloader = DataLoader(dataset_temporal, batch_size=batch_size, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader):

        graph = data

        assert graph.x.shape == (1024 * batch_size, 1)
        assert graph.target.shape == (1024 * batch_size, 3)
        assert graph.mask.shape == (1024 * batch_size, )

        break

def test_burgerpdedataset_fullsimu():

    # we choose the discretization of the space and the time
    nb_space = 1024
    nb_time = 201

    delta_x = 2.0 / nb_space
    delta_t = 1.0 / nb_time

    # we choose the batch size
    batch_size = 1

    # now we can create the dataloader
    edges, edges_index, mask = create_graph_burger(nb_space, delta_x, nb_nodes=None, nb_edges=None)

    path_hdf5 = "/app/data/1D_Burgers_Sols_Nu0.01.hdf5"

    # we create the dataset
    dataset = BurgerPDEDatasetFullSimulation(path_hdf5, edges, edges_index)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # we test the dataloader
    for i, data in enumerate(dataloader):

        assert data["nodes_t0"].shape == (batch_size, nb_space,3)

        assert data["image_result"].shape == (batch_size, nb_time, nb_space)

        assert data['nodes_boundary_x__1'].shape == (batch_size, nb_time,)
        assert data['nodes_boundary_x_1'].shape == (batch_size, nb_time,)

        break


def test_dataset_looking_into():

    path_hdf5 = "/app/data/1D_Burgers_Sols_Nu0.01.hdf5"

    with h5py.File(path_hdf5, "r") as f:
        print(f.keys())
        print(f["x-coordinate"].shape)
        print(f["t-coordinate"].shape)

    assert True


def test_create_darcy_graph():

    image = torch.zeros((4, 124, 124))

    delta_x = 1.0 / 124
    delta_y = 1.0 / 124

    edges_index, edges_attrib = create_darcy_graph(image.shape, delta_x, delta_y)

    assert edges_attrib.shape == (124 * 124 * 4 - 124 * 4, 2)
    assert edges_index.shape == (2, 124 * 124 * 4 - 124 * 4)

def test_darcy_dataset():

    path_hdf5 = "/app/data/2D_DarcyFlow_beta1.0_Train.hdf5"

    delta_x = 1.0 / 128
    delta_y = 1.0 / 128

    dataset = Darcy2DPDEDataset(path_hdf5, delta_x, delta_y)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader):

        graph = data

        assert graph.x.shape == (128 * 128, 3)
        assert graph.target.shape == (128 * 128, 1)

        assert graph.edge_attr.shape == (128 * 128 * 4 - 128 * 4, 2)
        assert graph.edge_index.shape == (2, 128 * 128 * 4 - 128 * 4)

        break