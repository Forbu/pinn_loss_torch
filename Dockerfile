# chargement de l'image Docker contenant pytorch et les drivers gpu (~3Go)
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

USER root
			
RUN apt update -y && \
    apt install -y build-essential && \
    apt install -y gcc && \
	DEBIAN_FRONTEND="noninteractive" TZ="Europe/Paris"

RUN which python3

# install pip 3.9
RUN apt install python3-pip -y

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# install pytorch_lightning
RUN pip install pytorch_lightning

# install mlflow (for metrics tracking)
# RUN pip install mlflow==1.27.0

# install pytest (for testing)
RUN pip install pytest

# install einops
RUN pip install einops

# install dvc (model versioning)
RUN pip install dvc

# install matplotlib (for plotting)
RUN pip install matplotlib

# install pckage to read hdf5 files
RUN pip install h5py

# install git without asking for confirmation or geographic location
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qq -y git
RUN git config --global user.name "Adrien B"
RUN git config --global user.email "forbu14@gmail.com"

# install wandb
RUN pip install wandb

# création des dossiers pour stocker les données
RUN mkdir /app
