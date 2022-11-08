import pytest 

from discretize_pinn_loss.pdebenchmark import BurgerPDEDataset, BurgerPDEDatasetFullSimulation
from discretize_pinn_loss.utils import create_graph_burger

# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader

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
        
        assert data["x"].shape == (batch_size*nb_space, 2)
        assert data["edge_attr"].shape == (batch_size*(nb_space - 1) * 2 , 1)
        assert data["edge_index"].shape == (2, batch_size*(nb_space - 1) * 2)

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

        assert data["nodes_t0"].shape == (batch_size, nb_space,2)

        assert data["image_result"].shape == (batch_size, nb_time, nb_space)

        assert data['nodes_boundary_x__1'].shape == (batch_size, nb_time,)
        assert data['nodes_boundary_x_1'].shape == (batch_size, nb_time,)

        break



