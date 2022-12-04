"""
Module where we use the PDE dataset to create our dataloader

"""

# import pytorch dataset and dataloader
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
import h5py
import numpy as np
import einops

from discretize_pinn_loss.pdebenchmark_darcy import create_darcy_graph

class NSCompressible2DPDEDataset(GeometricDataset):
    """
    This is a PDE dataset that we use to create our dataloader
    This is in graph mode using the torch geometric library
    """
    def __init__(self, path_hdf5, delta_x, delta_y):
        super(NSCompressible2DPDEDataset, self).__init__()

        self.path_hdf5 = path_hdf5
        self.delta_x = delta_x
        self.delta_y = delta_y

        with h5py.File(path_hdf5, "r") as f:
            self.batch_size, self.nb_time, self.img1, self.img2 = f["Vx"].shape

            self.t_coord = np.array(f["t-coordinate"])
            self.x_coord = np.array(f["x-coordinate"])
            self.y_coord = np.array(f["y-coordinate"])
            
        # TODO generate edges_attrib and edges_index
        self.edges_index, self.edges_attrib = create_darcy_graph((self.batch_size, self.img1, self.img2), self.delta_x, self.delta_y)
            
    def __len__(self):
        return self.batch_size * (self.nb_time - 3)

    def __getitem__(self, idx):

        # we get the index of the time step
        idx_t = idx % (self.nb_time - 3)

        # we get the index of the item
        idx_batch = idx // (self.nb_time - 3)

        # we open the hdf5 file
        with h5py.File(self.path_hdf5, "r") as f:
            
            # retrieve current velocity
            vx = np.array(f["Vx"][idx_batch, idx_t, :, :])
            vy = np.array(f["Vy"][idx_batch, idx_t, :, :])

            density = np.array(f["density"][idx_batch, idx_t, :, :])
            pressure = np.array(f["pressure"][idx_batch, idx_t, :, :])

            # retrieve next time step values
            vx_next = np.array(f["Vx"][idx_batch, idx_t + 1, :, :])
            vy_next = np.array(f["Vy"][idx_batch, idx_t + 1, :, :])

            density_next = np.array(f["density"][idx_batch, idx_t + 1, :, :])
            pressure_next = np.array(f["pressure"][idx_batch, idx_t + 1, :, :])



        # we reshape the data
        vx = einops.rearrange(vx, "h w -> (h w)")
        vy = einops.rearrange(vy, "h w -> (h w)")

        density = einops.rearrange(density, "h w -> (h w)").unsqueeze(1)
        pressure = einops.rearrange(pressure, "h w -> (h w)").unsqueeze(1)

        vx_next = einops.rearrange(vx_next, "h w -> (h w)")
        vy_next = einops.rearrange(vy_next, "h w -> (h w)")

        density_next = einops.rearrange(density_next, "h w -> (h w)").unsqueeze(1)
        pressure_next = einops.rearrange(pressure_next, "h w -> (h w)").unsqueeze(1)

        V = np.stack([vx, vy], axis=1)
        V_next = np.stack([vx_next, vy_next], axis=1)

        # we create the data object
        data = Data(
            V=torch.from_numpy(V).float(),
            V_next=torch.from_numpy(V_next).float(),
            density=torch.from_numpy(density).float(),
            pressure=torch.from_numpy(pressure).float(),
            density_next=torch.from_numpy(density_next).float(),
            pressure_next=torch.from_numpy(pressure_next).float(),
            edge_index=torch.from_numpy(self.edges_index).long(),
            edge_attr=torch.from_numpy(self.edges_attrib).float(),
        )
                    
        return data


        

