import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator, DarcyLoss
from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from torch.utils.data import DataLoader as DataLoader2

from discretize_pinn_loss.pdebenchmark_IFD import NSImcompressible2DPDEDataset

import torch

def xtest_pdebenchmark():
    """
    Simple test to try to load the dataset
    """

    path_data = "/app/data/ns_incom_inhom_2d_512-0.h5"
    delta_x = 1.0 / 512
    delta_y = 1.0 / 512
    
    delta_t = 5.0 / 1000

    dataset = NSImcompressible2DPDEDataset(path_data, delta_x=delta_x, delta_y=delta_y)

    # we test the dataloader
    for i, data in enumerate(dataset):
        print(data)
        assert data["velocity"].shape == (512*512, 2)
        break

    # we test the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    
    for i, data in enumerate(dataloader):
        assert data["velocity"].shape == (512*512, 2)
        break

def test_loss_IFD():
    path_data = "/app/data/ns_incom_inhom_2d_512-0.h5"
    delta_x = 1.0 / 512 
    delta_y = delta_x
    
    delta_t = 5.0 / 1000

    dataset = NSImcompressible2DPDEDataset(path_data, delta_x=delta_x, delta_y=delta_y)

    # we test the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    
    for i, data in enumerate(dataloader):
        assert data["velocity"].shape == (512*512, 2)
        break

    # we create the loss
    from discretize_pinn_loss.losses_IFD import IncompressibleFluidLoss

    loss_fn = IncompressibleFluidLoss(index_node_x=0, index_node_y=1, index_edge_x=0, index_edge_y=1,
                                                                 delta_x=delta_x)

    # we test the loss
    for i, data in enumerate(dataloader):
        velocity = data["velocity"]

        velocity_next = data["velocity_next"]

        force = data["force"]

        # create graph
        graph_v = Data(x=velocity_next, edge_index=data["edge_index"], edge_attr=data["edge_attr"])
        
        graph_v_previous = Data(x=velocity, edge_index=data["edge_index"], edge_attr=data["edge_attr"])

        loss_momentum_x, loss_momentum_y, loss_continuity = loss_fn(graph_v, graph_v_previous, None, mu=0.01, dt=delta_t, force=force)
        break


    print("loss_momentum_x", loss_momentum_x)
    print("loss_momentum_y", loss_momentum_y)

    print("loss_momentum_x", loss_momentum_x.abs().mean())
    print("loss_momentum_y", loss_momentum_y.abs().mean())
    print("loss_continuity", loss_continuity.abs().mean())

    # now we just want a little plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(velocity_next[:, 0].reshape(512, 512))
    plt.colorbar()
    
    # save
    plt.savefig("test_IFD.png")

    # now we just want a little plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(velocity[:, 0].reshape(512, 512))
    plt.colorbar()
    
    # save
    plt.savefig("test_IFD_bis.png")

    plt.figure()
    plt.imshow(velocity_next[:, 1].reshape(512, 512))
    plt.colorbar()

    # save
    plt.savefig("test_IFD2.png")

    plt.figure()
    plt.imshow(force[:, 0].reshape(512, 512))
    plt.colorbar()

    # save
    plt.savefig("test_IFD3.png")

    plt.figure()
    plt.imshow(force[:, 1].reshape(512, 512))
    plt.colorbar()

    plt.figure()
    plt.imshow(loss_continuity.reshape(512, 512))
    plt.colorbar()

    plt.title("loss_continuity")

    # save
    plt.savefig("test_IFD6.png")

    plt.figure()
    plt.imshow(loss_momentum_y.reshape(512, 512), vmin=-5, vmax=5)
    plt.colorbar()

    plt.title("loss_momentum_y")

    # save
    plt.savefig("test_IFD7.png")

    plt.figure()
    plt.imshow(loss_momentum_x.reshape(512, 512), vmin=-5, vmax=5)
    plt.colorbar()

    plt.title("loss_momentum_x")

    # save
    plt.savefig("test_IFD8.png")

    assert False


