"""
Module where we use the PDE dataset to create our dataloader

"""

# import pytorch dataset and dataloader
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np

class BurgerPDEDataset(Dataset):
    def __init__(self, path_hdf5, edges, edges_index, mask=None):
        self.path_hdf5 = path_hdf5
        self.edges = edges
        self.edges_index = edges_index
        self.mask = mask

        # first read to know the lenght of the dataset
        with h5py.File(self.path_hdf5, "r") as f:
            
            # we get the shape of f['tensor']
            tensor_arr = f['tensor']

            self.len_item = tensor_arr.shape[0]
            
            self.t = np.array(f['t-coordinate'])
            self.len_t = self.t.shape[0]

            self.x = np.array(f['x-coordinate'])

        if self.mask is None:
            self.mask = np.ones(self.x.shape)
            
    def __len__(self):
        return self.len_item * (self.len_t - 3)

    def __getitem__(self, idx):

        # we get the index of the time step
        idx_t = idx // (self.len_item - 3)

        # we get the index of the item
        idx_item = idx % self.len_item
        
        # we open the hdf5 file
        with h5py.File(self.path_hdf5, "r") as f:
            # we get the tensor
            tensor_arr = f['tensor']
            # we get the tensor at the index idx
            tensor = np.array(tensor_arr[idx_item, idx_t, :])

            # we concat the tensor with the mask
            tensor = np.concatenate([tensor.reshape((-1, 1)), self.mask], axis=1)

            tensor_tp1 = np.array(tensor_arr[idx_item, idx_t+1, :])

        return {"nodes" : tensor, "edges" : self.edges, "edges_index" : self.edges_index, "nodes_next" : tensor_tp1, "mask" : self.mask}

class BurgerPDEDatasetFullSimulation(Dataset):
    def __init__(self, path_hdf5, edges, edges_index, mask=None):
        self.path_hdf5 = path_hdf5
        self.edges = torch.Tensor(edges)
        self.edges_index = torch.Tensor(edges_index)

        self.mask = torch.Tensor(mask)

        # first read to know the lenght of the dataset
        with h5py.File(self.path_hdf5, "r") as f:
            
            # we get the shape of f['tensor']
            tensor_arr = f['tensor']

            self.len_item = tensor_arr.shape[0]
            
            self.t = np.array(f['t-coordinate'])
            self.len_t = self.t.shape[0]

            self.x = np.array(f['x-coordinate'])

        if self.mask is None:
            self.mask = np.ones(self.x.shape)
            
    def __len__(self):
        return self.len_item

    def __getitem__(self, idx):

        # we get the index of the item
        idx_item = idx 
        
        # we open the hdf5 file
        with h5py.File(self.path_hdf5, "r") as f:
            # we get the tensor
            tensor_arr = f['tensor']
            # we get the tensor at the index idx
            tensor = torch.Tensor(tensor_arr[idx_item, :, :])


            # here we should retrieve the initial condition
            tensor_t0 = torch.Tensor(tensor_arr[idx_item, 0, :])

            tensor_t0 = torch.cat([tensor_t0.reshape((-1, 1)), self.mask], axis=1)

            # boundary condition
            tensor_boundary_x__1 = torch.Tensor(tensor_arr[idx_item, :, 0])
            tensor_boundary_x_1 = torch.Tensor(tensor_arr[idx_item, :, -1])

        return {"image_result" : tensor, "edges" : self.edges, "edges_index" : self.edges_index,
                                 "nodes_t0" : tensor_t0, "nodes_boundary_x__1" : tensor_boundary_x__1, "nodes_boundary_x_1" : tensor_boundary_x_1}
