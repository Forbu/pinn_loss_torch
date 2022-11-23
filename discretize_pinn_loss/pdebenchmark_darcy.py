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

def create_darcy_graph(shape, delta_x, delta_y):
    """
    Function that return edges_attrib and edges_index for an image IxI
    :param shape: shape of the image
    :return: edges_attrib, edges_index
    
    """

    _, I, I = shape

    # we create the edge index
    edge_index_list = []

    # we create the edge attributes
    edge_attr_list = []

    # we create the node attributes
    # now we can init the node index
    node_index = torch.arange(I * I).reshape(I, I)

    for i in range(I):
        for j in range(I):
                
                # we get the index of the node
                index = node_index[i, j]

                for i_ in range(-1, 1 + 1):
                    for j_ in range(-1, 1 + 1):
                        """
                        Here we remove 5 cases:
                        - the case where i_ == 0 and j_ == 0
                        - the case where i_ == 1 and j_ == 1
                        - the case where i_ == -1 and j_ == -1
                        - the case where i_ == 1 and j_ == -1
                        - the case where i_ == -1 and j_ == 1
                        """

                        if (i_ == 0 or j_ == 0) and not (i_ == 0 and j_ == 0):

                            # check if the index is in the image
                            if (i + i_) >= 0 and (i + i_) < I and (j + j_) >= 0 and (j + j_) < I:

                                # we get the index of the neighbor
                                index_neighbor = node_index[i + i_, j + j_]

                                # we append the edge index
                                edge_index_list.append([index, index_neighbor])

                                # we append the edge attribute
                                edge_attr_list.append([i_ * delta_x, j_ * delta_y])    
                        


    # we create the edge index
    edge_index = torch.tensor(edge_index_list).T

    # we create the edge attributes
    edge_attr = torch.tensor(edge_attr_list)

    return edge_index, edge_attr


def init_solution(shape, xrange=[0, 1], yrange=[0, 1], node=True):
    """
    Function that return the initial solution for a given shape
    :param shape: shape of the image
    :return: initial solution
    """
    b, I, J = shape
    mid_x = (xrange[1] - xrange[0]) / 2
    mid_y = (yrange[1] - yrange[0]) / 2

    # we create the initial solution
    initial_solution = torch.zeros(I, J, 1)

    # we create the x and y coordinates
    x = torch.linspace(xrange[0], xrange[1], I)

    y = torch.linspace(yrange[0], yrange[1], J)

    # we create the meshgrid
    x, y = torch.meshgrid(x, y)

    # we create the initial solution (lorentz function)
    initial_solution[:, :, 0] = 1 / (5 + 20*(x - mid_x) ** 2 + 20*(y - mid_y) ** 2)

    if node:
        # now we can flatten the initial solution
        initial_solution = initial_solution.reshape(-1, 1)

        # now we reapet the initial solution b times
        initial_solution = initial_solution.repeat(b, 1)
    else:
        initial_solution = initial_solution.unsqueeze(0).repeat(b, 1, 1, 1)

    return initial_solution


class Darcy2DPDEDataset(GeometricDataset):
    """
    This is a PDE dataset that we use to create our dataloader
    This is in graph mode using the torch geometric library
    """
    def __init__(self, path_hdf5, delta_x, delta_y):
        super(Darcy2DPDEDataset, self).__init__()

        self.path_hdf5 = path_hdf5
        self.delta_x = delta_x
        self.delta_y = delta_y

        with h5py.File(path_hdf5, "r") as f:

            # we get the tensor
            tensor_arr = f['tensor']

            self.shape_data = tensor_arr.shape
            self.shape_data = (self.shape_data[0], self.shape_data[2], self.shape_data[3])

            # we get the tensor at the index idx
            self.x = np.array(f['x-coordinate'])

            # we get the tensor at the index idx
            self.y = np.array(f['y-coordinate'])

        # TODO generate edges_attrib and edges_index
        self.edges_index, self.edges_attrib = create_darcy_graph(self.shape_data, self.delta_x, self.delta_y)
            
    def __len__(self):
        return self.shape_data[0]

    def __getitem__(self, idx):

        # we open the hdf5 file
        with h5py.File(self.path_hdf5, "r") as f:
            # we get the tensor
            tensor_arr = f['nu']
            # we get the tensor at the index idx
            tensor = np.array(tensor_arr[idx, :, :])

            target = np.array(f['tensor'][idx, 0, :, :])
    
            # create the mask for limit condition
            mask = np.zeros_like(target)

            # we set the mask to 1 for the limit condition
            mask[0, :] = 1
            mask[-1, :] = 1
            mask[:, 0] = 1
            mask[:, -1] = 1

            limit_condition = np.where(mask == 1, target, 0)

            node = einops.rearrange(tensor, 'h w -> (h w)')
            target = einops.rearrange(target, 'h w -> (h w)')
            limit_condition = einops.rearrange(limit_condition, 'h w -> (h w)')
            mask = einops.rearrange(mask, 'h w -> (h w)')

            # we convert everything to torch tensor
            node = torch.tensor(node, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
            limit_condition = torch.tensor(limit_condition, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

            # we add one dimension at the end for the 4 channels
            node = node.unsqueeze(1)
            target = target.unsqueeze(1)
            limit_condition = limit_condition.unsqueeze(1)
            mask = mask.unsqueeze(1)

        # concat node mask and limit condition as input
        input_ = torch.cat([node, mask, limit_condition], dim=1)

        # we create the data
        data = Data(x=torch.tensor(input_, dtype=torch.float), target=torch.tensor(target, dtype=torch.float),
                    edge_index=torch.tensor(self.edges_index, dtype=torch.long),
                    edge_attr=torch.tensor(self.edges_attrib, dtype=torch.float), mask=mask.squeeze())
                    
        return data
        

