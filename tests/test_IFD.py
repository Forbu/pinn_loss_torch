import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator, DarcyLoss
from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator
from torch_geometric.data import Data

from discretize_pinn_loss.pdebenchmark_darcy import create_darcy_graph, init_solution

import torch

