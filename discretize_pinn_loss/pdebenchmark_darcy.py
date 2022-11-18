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

class Darcy2DPDEDataset(GeometricDataset):
    """
    This is a PDE dataset that we use to create our dataloader
    This is in graph mode using the torch geometric library
    """
    def __init__(self, path_hdf5):
        super(Darcy2DPDEDataset, self).__init__()

        self.path_hdf5 = path_hdf5

        with h5py.File(path_hdf5, "r") as f:

            # we get the tensor
            tensor_arr = f['tensor']

            self.shape_data = tensor_arr.shape

            # we get the tensor at the index idx
            self.x = np.array(f['x-coordinate'])

            # we get the tensor at the index idx
            self.y = np.array(f['y-coordinate'])

        # TODO generate edges_attrib and edges_index
        self.edges_index = None
        self.edges_attrib = None
            
    def __len__(self):
        return self.shape_data[0]

    def __getitem__(self, idx):

        # we open the hdf5 file
        with h5py.File(self.path_hdf5, "r") as f:
            # we get the tensor
            tensor_arr = f['tensor']
            # we get the tensor at the index idx
            tensor = np.array(tensor_arr[idx, 0, :, :])

            # flatten the tensor to get the node
            node = einops.rearrange(tensor, 'h w -> (h w)')


        # we create the data
        data = Data(x=torch.tensor(node, dtype=torch.float),
                    edge_index=torch.tensor(self.edges_index, dtype=torch.long),
                    edge_attr=torch.tensor(self.edges_attrib, dtype=torch.float))
                    

        return data
        

