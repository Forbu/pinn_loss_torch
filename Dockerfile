# dockerfile definition to create a docker image for the application (pinn_loss_gnn)
# jax base image
FROM nvidia/cuda:11.3.0-runtime-ubuntu22.04

# install python3.9
RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qq -y python3.9 python3.9-dev python3.9-distutils

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install pytorch geometric
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# install mlflow (for metrics tracking)
RUN pip install mlflow==1.27.0

# install pytest (for testing)
RUN pip install pytest

# install dvc (model versioning)
RUN pip install dvc

# install torch (only to use the dataloder / dataset capabilities)
RUN pip install torch

# install matplotlib (for plotting)
RUN pip install matplotlib

# install pckage to read hdf5 files
RUN pip install h5py

# install git without asking for confirmation or geographic location
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qq -y git


RUN git config --global user.name "Adrien B"

RUN git config --global user.email "forbu14@gmail.com"
