
# init setup.py
from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='discretize_pinn_loss',
    version='0.1',
    description='A package for PINN loss with graph neural networks',
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    ## add data file to package file : discretize_pinn_loss\data_init\init_solution.pt
    package_data={'discretize_pinn_loss': ['data_init/init_solution.pt']},
    data_files=[('discretize_pinn_loss', ['discretize_pinn_loss/data_init/init_solution.pt'])],
)