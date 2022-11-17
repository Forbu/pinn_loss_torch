FROM julia:latest

# Install python3 and pip3
RUN apt-get update && apt-get install -y python3 python3-pip

# Install python packages
RUN pip3 install numpy pandas matplotlib seaborn

# install pySR
RUN pip3 install -U pysr

RUN python3 -c 'import pysr; pysr.install()'

# install pckage to read hdf5 files
RUN pip3 install h5py
