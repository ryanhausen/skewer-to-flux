# Skewer to Flux - Starter

This repo is to help you get the data you need to start prototyping models.

To get started create a new python environment and install the dependencies
in the requirements file.

`pip install -r requirements.txt`

The download a subset of the data to work with:

`python download_labels.py`

This will download a file called `data_ys.npy` it contains 60,000 simulation
skewers, which we are going to predict from the flux. To generate the flux
values run:

`python generate_data.py`

This will create a file called `data_xs.npy`, which will correspond to the
values in `data_ys.npy`. After the files are downloaded you can open up
`DataExploration.ipynb` and see a simple example of plotting one sample.