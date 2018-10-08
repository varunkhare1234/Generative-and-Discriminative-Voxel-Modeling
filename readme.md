# Generative-and-Discriminative-Voxel-Modeling
Voxel-Based Convnets for Classification

This is the pytorch implementation of the voxception-resnet as described in the [original paper](https://arxiv.org/pdf/1608.04236.pdf)

## Preparing the data
For the Discriminative model, I've included a MATLAB script in utils to convert raw Modelnet .off files into MATLAB arrays, then a python script to convert the MATLAB arrays into either .npz files or hdf5 files (for use with [fuel](https://github.com/mila-udem/fuel)) or directly as dataset. 

The _nr.tar files contain the unaugmented Modelnet10 train and test sets, while the other tar files have 12 copies of each model, rotated evenly about the vertical axis. 

## Training a Classifier
the code for training and testinf is present in the file test_and_train.py while the model itself is present in voxception_resnet_pytorch.py
## Evaluating an Ensemble
#
You can produce a simple ensemble by averaging multiple models' predictions on the test sets. I provide six pre-trained models for this purpose, along with .csv files containing their outputs on ModelNet40.
Use the test_ensemble.py script to produce a .csv file with the model's predictions, and use the ensemble.m MATLAB script to combine and evaluate all the results.

