import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator, DarcyLoss
from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from torch.utils.data import DataLoader as DataLoader2

from discretize_pinn_loss.pdebenchmark_CFD import NSCompressible2DPDEDataset

import matplotlib.pyplot as plt

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
        
        # we retrieve the data and make a graph
        V = data["V"]
        P = data["pressure"]
        d = data["density"]

        V_next = data["V_next"]
        P_next = data["pressure_next"]
        d_next = data["density_next"]

        # we create the graph
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]

        # we create the graph
        graph_v = Data(x=V, edge_index=edge_index, edge_attr=edge_attr) 
        graph_p = Data(x=P, edge_index=edge_index, edge_attr=edge_attr)
        graph_d = Data(x=d, edge_index=edge_index, edge_attr=edge_attr)

        # we create the graph
        graph_v_next = Data(x=V_next, edge_index=edge_index, edge_attr=edge_attr)
        graph_p_next = Data(x=P_next, edge_index=edge_index, edge_attr=edge_attr)
        graph_d_next = Data(x=d_next, edge_index=edge_index, edge_attr=edge_attr)




        break

    V = V.reshape(512, 512, 2)
    P = P.reshape(512, 512, 1)
    d = d.reshape(512, 512, 1)

    V_next = V_next.reshape(512, 512, 2)
    P_next = P_next.reshape(512, 512, 1)
    d_next = d_next.reshape(512, 512, 1)

    # now we manually compute the continuity equation
    # we compute the divergence of the velocity
    d_V = V * d

    # we compute the divergence of the velocity
    div_d_V = torch.zeros(512, 512, 1)

    div_d_V[1:-1, 1:-1, 0] = (d_V[2:, 1:-1, 0] - d_V[:-2, 1:-1, 0]) / (2 * delta_x) + (d_V[1:-1, 2:, 1] - d_V[1:-1, :-2, 1]) / (2 * delta_y)

    # now we compute the density derivative
    deriv_d = (d_next - d) / delta_t

    # we compute the continuity equation
    continuity = deriv_d + div_d_V

    # now we plot the result
    plt.imshow(continuity.reshape(512, 512))
    plt.colorbar()
    plt.savefig("continuity.png")

    assert False


    # we compute the loss
    loss = loss_fn(graph_v_next, graph_v, graph_p_next, graph_p, graph_d_next, graph_d,  M=0.1, eta=1e-08, zeta=1e-08, dt=delta_t)         

    print(loss.abs().mean())


    # reshape the loss
    loss = loss.reshape((512, 512, 1))

    # plot
    

    plt.imshow(loss.detach().numpy())
    plt.colorbar()
    plt.savefig("loss.png")

    # save

    assert False


