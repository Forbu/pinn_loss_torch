import pytest


from discretize_pinn_loss.losses_darcy import Nabla2DOperator, Nabla2DProductOperator, DarcyFlowOperator, DarcyLoss
from discretize_pinn_loss.loss_operator import SpatialDerivativeOperator
from torch_geometric.data import Data

from discretize_pinn_loss.pdebenchmark_darcy import create_darcy_graph, init_solution

import torch

def test_nabla2d_operator():
    # we create the operator
    nabla2d_operator = Nabla2DOperator(delta_x=1, delta_y=1)

    x = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, 2)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d = nabla2d_operator(graph)

    assert nabla2d.shape == (100, 2)

def test_nabla2d_product_operator():

    # we create the operator
    nabla2d_product_operator = Nabla2DProductOperator(delta_x=1, delta_y=1)

    x = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, 2)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d_product = nabla2d_product_operator(graph)

    assert nabla2d_product.shape == (100, )

def test_darcy_flow_operator():

    # we create the operator
    darcy_flow_operator = DarcyFlowOperator(delta_x=1, delta_y=1)

    x = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 100))
    edge_attr = torch.randn(100, 2)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    a_x = torch.randn(100, 1)
    f = 4

    # create the graph
    graph_a_x = Data(x=a_x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    darcy_flow = darcy_flow_operator(graph, graph_a_x, f)

    assert darcy_flow.shape == (100, )

def test_darcy_flow_operator_2():
    """
    We try to test backward pass on the dartcy flow operator
    """
    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # we create the operator
    darcy_flow_operator = DarcyFlowOperator(delta_x=delta_x, delta_y=delta_y)

    x = torch.randn(nb_nodes, 2, requires_grad=True)/10.
    x = torch.nn.Parameter(x)

    # require grad for x

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    a_x = torch.randn(nb_nodes, 1, requires_grad=True)
    graph_a_x = Data(x=a_x, edge_index=edge_index, edge_attr=edge_attr)

    f = 1.

    # we compute the nabla2d
    darcy_flow = darcy_flow_operator(graph, graph_a_x, f)

    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(darcy_flow, torch.zeros_like(darcy_flow))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    assert not torch.isnan(x.grad).any()

    assert x.grad is not None

def test_spatial_backward():
    """
    We try to test backward pass on the dartcy flow operator
    combined with FNO operator
    """
    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # we create the operator
    nabla2d_operator = SpatialDerivativeOperator(index_derivative_edge=0, index_derivative_node=0)

    x = torch.randn(nb_nodes, 2, requires_grad=True)

    # we remove edge where edge_attr is 0
    edge_index = edge_index[:, edge_attr[:, 0] != 0]
    edge_attr = edge_attr[edge_attr[:, 0] != 0]

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d = nabla2d_operator(graph)

    print("nabla2d", nabla2d.shape)

    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(nabla2d, torch.zeros_like(nabla2d))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    assert not torch.isnan(x.grad).any()

    assert x.grad is not None


def test_nabla_backward():
    """
    We try to test backward pass on the dartcy flow operator
    combined with FNO operator
    """
    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # we create the operator
    nabla2d_operator = Nabla2DOperator(delta_x, delta_y, index_derivative_node=0,
                                             index_derivative_x=0, index_derivative_y=1)

    x = torch.randn(nb_nodes, 2, requires_grad=True)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d = nabla2d_operator(graph)

    print("nabla2d", nabla2d.shape)

    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(nabla2d, torch.zeros_like(nabla2d))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    assert not torch.isnan(x.grad).any()
    assert x.grad is not None

def test_nabla2dProduct_backward():

    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # we create the operator
    nabla2d_product_operator = Nabla2DProductOperator(delta_x, delta_y, index_derivative_node=0,
                                             index_derivative_x=0, index_derivative_y=1)

    x = torch.randn(nb_nodes, 2, requires_grad=True)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d
    nabla2d_product = nabla2d_product_operator(graph)

    print("nabla2d_product", nabla2d_product.shape)

    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(nabla2d_product, torch.zeros_like(nabla2d_product))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    print(x.grad)

    # check if their is no any nan
    assert not torch.isnan(x.grad).any()

    assert x.grad is not None

def test_fulldarcy_backward():
    shape = (1, 128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    edge_index, edge_attr = create_darcy_graph(shape, delta_x, delta_y)

    nb_nodes = 128*128

    # loss and check if their is no any nan
    loss_fn = torch.nn.MSELoss()

    x = torch.randn(nb_nodes, 2, requires_grad=True)
    x = x/10.

    # require grad for x
    x = torch.nn.Parameter(x)
    

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    a_x = torch.randn(nb_nodes, 1, requires_grad=True)

    # init nabla2d operator

    nabla2d_operator = Nabla2DOperator(delta_x, delta_y, index_derivative_node=0,
                                        index_derivative_x=0, index_derivative_y=1)

    # we compute the nabla2d
    nabla2d = nabla2d_operator(graph)

    print("nabla2d", nabla2d)

    # we compute the nabla2d product
    nabla2d_product_operator = Nabla2DProductOperator(delta_x, delta_y, index_derivative_node=0,    
                                                        index_derivative_x=0, index_derivative_y=1) 


    tmp_product_a_x_nabla2d = nabla2d * a_x

    print("tmp_product_a_x_nabla2d", tmp_product_a_x_nabla2d)

    # we create the graph
    graph_tmp_product_a_x_nabla2d = Data(x=tmp_product_a_x_nabla2d, edge_index=edge_index, edge_attr=edge_attr)

    # we compute the nabla2d product
    nabla2d_product = nabla2d_product_operator(graph_tmp_product_a_x_nabla2d)

    print("nabla2d_product", nabla2d_product)

    # loss and check if their is no any nan
    loss_fn = torch.nn.MSELoss()

    # we compute the loss (mse)
    loss = loss_fn(nabla2d_product, torch.zeros_like(nabla2d_product))

    print("loss: ", loss)

    # we compute the gradient
    loss.backward()

    print(x.grad)

    # check if their is no any nan
    assert not torch.isnan(x.grad).any()

    assert x.grad is not None


def test_darcy_flow_operator_convergence():
    """
    We try to test backward pass on the dartcy flow operator
    """
    shape = (128, 128)

    delta_x = 1./128.
    delta_y = 1./128.

    from discretize_pinn_loss.pdebenchmark_darcy import Darcy2DPDEDataset
    from torch_geometric.data import Data, DataLoader

    path = "/app/data/2D_DarcyFlow_beta1.0_Train.hdf5"

    # init dataset and dataloader
    dataset = Darcy2DPDEDataset(path_hdf5=path, delta_x=delta_x, delta_y=delta_y)

    # divide into train and test
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # get the first batch of dataloader_test
    a_x = test_dataset[2]

    # we take the fist sample
    # a_x = dataset[0] 

    # we create the operator
    darcy_flow_operator = DarcyLoss(delta_x=delta_x, delta_y=delta_y)

    nb_nodes = 128*128

    x = a_x.target.clone()
    #x = torch.randn(nb_nodes, 1, requires_grad=True)/1000.
    x = init_solution((1, 128, 128))
    # init x with tensor file from discretize_pinn_loss\data_init\init_solution.pt
    x = torch.load("/app/discretize_pinn_loss/data_init/init_solution.pt")

    # flatten x
    x = x.view(-1, 1)

    x = torch.nn.Parameter(x)

    # init optimizer
    optimizer = torch.optim.Adam([x], lr=1e-2)

    # import MultiStepLR
    from torch.optim.lr_scheduler import MultiStepLR

    scheduler2 = MultiStepLR(optimizer, milestones=[100, 1000, 2000, 3000, 4000], gamma=0.3)

    # zero grad
    optimizer.zero_grad()

    nb_iter = 10

    loss_fn = torch.nn.MSELoss()

    lost_history = []

    for _ in range(nb_iter):
        
        y = x * (1 - a_x.mask.reshape(-1, 1))

        graph = Data(x=y, edge_index=a_x.edge_index, edge_attr=a_x.edge_attr)
        graph_a_x = a_x

        # we compute the nabla2d
        darcy_flow = darcy_flow_operator(graph, graph_a_x, a_x.mask)

        # we compute the loss (mse)
        loss = loss_fn(darcy_flow, torch.zeros_like(darcy_flow))

        # append to history
        lost_history.append(loss.detach().cpu().item())

        print("loss: ", loss)

        # we compute the gradient
        loss.backward()

        # step on the optimizer
        optimizer.step()
        scheduler2.step()

        # zero grad
        optimizer.zero_grad()


    # reshape x to get to the image shape
    x = x.reshape(shape)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(lost_history)
    plt.ylim(0, 1000)
    
    # save the image
    plt.savefig("loss_history.png")

    graph_target = Data(x=a_x.target, edge_index=a_x.edge_index, edge_attr=a_x.edge_attr)

    darcy_flow_target = darcy_flow_operator(graph_target, graph_a_x, a_x.mask)

    loss_target = loss_fn(darcy_flow_target, torch.zeros_like(darcy_flow_target))

    print("loss_target: ", loss_target)

    target = a_x.target.reshape(shape)

    a_x_row = a_x.x[:, 0].reshape(shape)

    # compute vmin and vman with the target
    vmin = target.min()
    vmax = target.max()

    # compute difference between target and x
    diff = target - x

    # create subplot for 2 figures : target and prediction
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    
    # plot target
    ax[0].imshow(target.detach().numpy(), cmap="jet", vmin=vmin, vmax=vmax)
    ax[0].set_title("target")

    # plot prediction
    ax[1].imshow(x.detach().numpy(), cmap="jet", vmin=vmin, vmax=vmax)
    ax[1].set_title("prediction")

    # plot difference
    ax[2].imshow(diff.detach().numpy(), cmap="jet", vmin=diff.min(), vmax=diff.max())
    ax[2].set_title("difference")

    # plot a_x
    ax[3].imshow(a_x_row.detach().numpy(), cmap="jet", vmin=a_x_row.min(), vmax=a_x_row.max())
    ax[3].set_title("a_x")

    print("vmin: ", diff.min())
    print("vmax: ", diff.max())

    # show colorbar
    fig.colorbar(ax[1].imshow(x.detach().numpy(), cmap="jet", vmin=vmin, vmax=vmax), ax=ax.ravel().tolist())

    # save image into png
    plt.savefig("test_darcy_flow_operator_convergence.png")

    assert True

def test_temporal_convergence():
    """
    script to test the temporal convergence of the darcy flow operator
    """
    shape = (128, 128)

    delta_x = 1./128.
    delta_y = 1./128.
    delta_t = 0.01

    from discretize_pinn_loss.pdebenchmark_darcy import Darcy2DPDEDataset
    from torch_geometric.data import Data, DataLoader

    path = "/app/data/2D_DarcyFlow_beta1.0_Train.hdf5"

    # init dataset and dataloader
    dataset = Darcy2DPDEDataset(path_hdf5=path, delta_x=delta_x, delta_y=delta_y)

    # divide into train and test
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # get the first batch of dataloader_test
    a_x = test_dataset[2]

    # we take the fist sample
    # a_x = dataset[0] 

    # we create the operator
    darcy_flow_operator = DarcyLoss(delta_x=delta_x, delta_y=delta_y)

    loss_fn = torch.nn.MSELoss()

    nb_nodes = 128*128

    x = a_x.target.clone()
    #x = torch.randn(nb_nodes, 1, requires_grad=True)/1000.
    x = init_solution((1, 128, 128))*0. + 1.
    x = a_x.target.clone()

    nb_time_step = 10

    # init optimizer

    error = 1.

    t_current = x.clone()
    t_current = t_current * (1 - a_x.mask.reshape(-1, 1))
    t_current = t_current.detach().clone()

    for idx_time in range(nb_time_step):

        # init optimizer
        t_new = t_current.clone()
        t_new = t_new.detach().requires_grad_(True)
        t_new = torch.nn.Parameter(t_new)
        optimizer = torch.optim.Adam([t_new], lr=1e-1)

        error = 100.

        nb_iter_optim = 0

        while error > 1 and nb_iter_optim < 100:

            error = torch.norm(t_new - t_current).item()
        
            y = t_new * (1 - a_x.mask.reshape(-1, 1))

            graph = Data(x=y, edge_index=a_x.edge_index, edge_attr=a_x.edge_attr)
            graph_a_x = a_x

            # we compute the nabla2d
            darcy_flow_loss = darcy_flow_operator(graph, graph_a_x, a_x.mask)
            temporal_loss = (y - t_current) / delta_t

            # we compute the loss (mse)
            loss = darcy_flow_loss + temporal_loss
            loss = loss_fn(loss, torch.zeros_like(loss))

            error = loss


            # we compute the gradient
            loss.backward()

            # step on the optimizer
            optimizer.step()

            # zero grad
            optimizer.zero_grad()

            nb_iter_optim += 1


        print("loss: ", loss)
        print("nb_iter_optim: ", nb_iter_optim)

        t_current = t_new.clone()
        t_current = t_current * (1 - a_x.mask.reshape(-1, 1))
        t_current = t_current.detach()

    print("t_current: ", t_current.shape)

    # reshape x to get to the image shape
    t_current = t_current.reshape(shape)

    print("t_current: ", t_current.shape)

    import matplotlib.pyplot as plt

    graph_target = Data(x=a_x.target, edge_index=a_x.edge_index, edge_attr=a_x.edge_attr)

    darcy_flow_target = darcy_flow_operator(graph_target, graph_a_x, a_x.mask)

    loss_target = loss_fn(darcy_flow_target, torch.zeros_like(darcy_flow_target))

    print("loss_target: ", loss_target)

    target = a_x.target.reshape(shape)

    a_x_row = a_x.x[:, 0].reshape(shape)

    # compute vmin and vman with the target
    vmin = target.min()
    vmax = target.max()

    # compute difference between target and x
    diff = target - t_current

    # create subplot for 2 figures : target and prediction
    fig, ax = plt.subplots(1, 4, figsize=(10, 10))
    
    # plot target
    ax[0].imshow(target.detach().numpy(), cmap="jet", vmin=vmin, vmax=vmax)
    ax[0].set_title("target")

    print("t_current", t_current.detach().numpy().shape)
    print("target", t_current.detach().numpy())

    # plot prediction
    ax[1].imshow(t_current.detach().numpy(), cmap="jet", vmin=vmin, vmax=vmax)
    ax[1].set_title("prediction")

    print(diff.shape)

    # plot difference
    ax[2].imshow(diff.detach().numpy(), cmap="jet", vmin=diff.min(), vmax=diff.max())
    ax[2].set_title("difference")

    # plot a_x
    ax[3].imshow(a_x_row.detach().numpy(), cmap="jet", vmin=a_x_row.min(), vmax=a_x_row.max())
    ax[3].set_title("a_x")

    print("vmin: ", diff.min())
    print("vmax: ", diff.max())

    # show colorbar
    fig.colorbar(ax[1].imshow(x.detach().numpy(), cmap="jet", vmin=vmin, vmax=vmax), ax=ax.ravel().tolist())

    # save image into png
    plt.savefig("test_darcy_flow_operator_convergence_temporal.png")

    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        # plot prediction
    ax[1].imshow(t_current.detach().numpy(), cmap="jet", vmin=vmin, vmax=vmax)
    ax[1].set_title("prediction")

    # save image into png
    plt.savefig("test_darcy_flow_operator_convergence_temporal_2.png")


    assert False