import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator, DarcyLoss
from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from torch.utils.data import DataLoader as DataLoader2

from discretize_pinn_loss.pdebenchmark_CFD import NSCompressible2DPDEDataset

import torch

def test_pdebenchmark():
    """
    Simple test to try to load the dataset
    """

    path_data = "/app/data/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
    delta_x = 1.0 / 512
    delta_y = 1.0 / 512
    
    delta_t = 5.0 / 1000

    dataset = NSCompressible2DPDEDataset(path_data, delta_x=delta_x, delta_y=delta_y)

    # we test the dataloader
    for i, data in enumerate(dataset):
        print(data)
        assert data["V"].shape == (512*512, 2)
        break

    # we test the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    
    for i, data in enumerate(dataloader):
        assert data["V"].shape == (512*512, 2)
        break

def test_loss_IFD():
    path_data = "/app/data/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
    delta_x = 1.0 / 512 
    delta_y = delta_x
    
    delta_t = 0.05

    dataset = NSCompressible2DPDEDataset(path_data, delta_x=delta_x, delta_y=delta_y)

    # we test the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    
    for i, data in enumerate(dataloader):
        assert data["V"].shape == (512*512, 2)
        break

    # we create the loss
    from discretize_pinn_loss.losses_CFD import CompressibleFluidLoss

    loss_fn = CompressibleFluidLoss(index_node_x=0, index_node_y=1, index_edge_x=0, index_edge_y=1,
                                                                 delta_x=delta_x)

    # we test the loss
    for i, data in enumerate(dataloader):
        break


    assert False


