# Generative-and-Discriminative-Voxel-Modeling
Voxel-Based Convnets for Classification

![GUI](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modelling/blob/master/doc/GUI3.png)
This is the pytorch implementation of the voxception-resnet as described in the original paper

## Preparing the data
I've included several .tar versions of Modelnet10, which can be used to train the VAE and run the GUI. If you wish to write more .tar files (say, of Modelnet40) for use with the VAE and GUI, download  [the dataset](http://modelnet.cs.princeton.edu/) and then see [voxnet](https://github.com/dimatura/voxnet).

For the Discriminative model, I've included a MATLAB script in utils to convert raw Modelnet .off files into MATLAB arrays, then a python script to convert the MATLAB arrays into either .npz files or hdf5 files (for use with [fuel](https://github.com/mila-udem/fuel)). 

The _nr.tar files contain the unaugmented Modelnet10 train and test sets, while the other tar files have 12 copies of each model, rotated evenly about the vertical axis. 

## Training a Classifier
The VRN.py file contains the model configuration and definitions for any custom layer types. The model can be trained with:

```sh
python Discriminative/train.py Discriminative/VRN.py datasets/modelnet40_rot_train.npz
```
Note that running the train function will start from scratch and overwrite any pre-trained model weights (.npz files with the same name as their corresponding config files). Use the --resume=True option to resume training from an earlier session or from one of the provided pre-trained models.

The first time you compile these functions may take a very long time, and may exceed the maximum recursion depth allowed by python.


## Evaluating an Ensemble
#
You can produce a simple ensemble by averaging multiple models' predictions on the test sets. I provide six pre-trained models for this purpose, along with .csv files containing their outputs on ModelNet40.
Use the test_ensemble.py script to produce a .csv file with the model's predictions, and use the ensemble.m MATLAB script to combine and evaluate all the results.

## Acknowledgments
This code was originally based on [voxnet](https://github.com/dimatura/voxnet) by D. Maturana.
