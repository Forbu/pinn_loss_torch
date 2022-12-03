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

class NSImcompressible2DPDEDataset(GeometricDataset):
    """
    This is a PDE dataset that we use to create our dataloader
    This is in graph mode using the torch geometric library
    """
    def __init__(self, path_hdf5, delta_x, delta_y):
        super(NSImcompressible2DPDEDataset, self).__init__()

        self.path_hdf5 = path_hdf5
        self.delta_x = delta_x
        self.delta_y = delta_y

        with h5py.File(path_hdf5, "r") as f:
            self.batch_size, self.nb_time, self.img1, self.img2, _ = f["velocity"].shape

            self.force = np.array(f["force"])

        # TODO generate edges_attrib and edges_index
        self.edges_index, self.edges_attrib = create_darcy_graph((self.batch_size, self.img1, self.img2), self.delta_x, self.delta_y)
            
    def __len__(self):
        return self.batch_size * (self.nb_time - 3)

    def __getitem__(self, idx):

        # we get the index of the time step
        idx_t = idx // (self.batch_size - 3)

        # we get the index of the item
        idx_item = idx % self.batch_size

        # we open the hdf5 file
        with h5py.File(self.path_hdf5, "r") as f:
            
            # retrieve current velocity
            velocity = np.array(f["velocity"][idx_item, idx_t, :, :, :])

            # retrieve current force
            force = np.array(f["force"][idx_item, :, :, :])

            # retrieve next velocity
            velocity_next = np.array(f["velocity"][idx_item, idx_t + 1, :, :, :])

        # we create the data
        data = Data(velocity=torch.from_numpy(velocity).float(),
                    force=torch.from_numpy(force).float(),
                    velocity_next=torch.from_numpy(velocity_next).float(),
                    edges_index=torch.from_numpy(self.edges_index).long(),
                    edges_attrib=torch.from_numpy(self.edges_attrib).float())
                    
        return data


        

