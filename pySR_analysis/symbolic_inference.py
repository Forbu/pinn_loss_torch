from pysr import PySRRegressor
import numpy as np

import h5py

path_hdf5 = "/app/data/1D_Burgers_Sols_Nu0.01.hdf5"

batch_size = 10
delta_x = 1./1024
delta_t = 2./200

with h5py.File(path_hdf5, "r") as f:

    tensor_arr = f['tensor'] # tensor is a 3D array of size (1000, 201, 1024)

    # we select the 10th first sample
    tensor = np.array(tensor_arr[:batch_size, :, :], dtype=np.float32)

    t = np.array(f['t-coordinate']) # t is a 1D array of size (201,)
    x = np.array(f['x-coordinate']) # x is a 1D array of size (1024,)

print(x)

# now we want to rework the tensor in order to get a 2D array of size (201 * 10 * 1024//5, 5 + 1)
# where the first 5 columns are the 5 first values of the 10 samples at each time step
# and the last column is the 3th value of the 10 samples at each time step
X = []
Y = []

for i in range(batch_size):
    for j in range(0, 200):
        for k in range(0, 1024 - 6):
            X.append(tensor[i, j, k:k+5])
            Y.append(tensor[i, j+1, k+2] - tensor[i, j, k+2])

X = np.array(X)
Y = np.array(Y)

# we reshape the array
X = X.reshape((-1, 5))
Y = Y.reshape((-1, 1))

# if filter our dataset if Y == 0
mask = Y[:, 0] != 0
X = X[mask, :]
Y = Y[mask, :]

# we select only 9000 samples
X = X[:9000]
Y = Y[:9000]

# we add delta_x and delta_t to X
# X = np.concatenate((X, delta_x * np.ones((X.shape[0], 1)), delta_t * np.ones((X.shape[0], 1))), axis=1)

# convert into float32
X = X.astype(np.float32)
Y = Y.astype(np.float32)

# we create the regressor
regressor = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=200,
    binary_operators=["+", "*", "max", "min", "/"],
    unary_operators=[
        # julia relu function
        "inv(x) = 1/x",
	# ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1. / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
)

# we fit the regressor
regressor.fit(X, Y)

# we print the result
print(regressor)
        
