# Image Classifier: Udacity Nanodegree - AI Programming with Python Final Project

An image classifier that identifies flower types in an image. I created this to learn numpy, matplotlib, jupyter notebooks, neural networks and machine learning with pytorch. It's the final project in the Udacity Nanodegree: AI Programming with Python.

## Getting Started With Development

### Dependencies
- Python 3.6.6
- pytorch==0.4.0
- torchvision==0.2.1

### Setup
- Download [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and unpack into the images directory using the [torchvision datasets format](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder). See [images folder README](https://github.com/gregdferrell/aipy-p1-image-classifier/blob/master/images/README.md) for more details.

## Running the App

#### train.py
`train.py` is a script to train a neural network on the given data directory. Use command-line options to customize the
hyperparameters of the neural network.

Prints out training loss, validation loss, and validation accuracy as the network trains. Then saves a checkpoint.

To see command-line options, exec: `python train.py --help`

Example execution and result:
```
$ python train.py flowers --gpu --epochs=1

Training ...
Epoch: 1/1..  Training Loss: 3.334..  Test Loss: 1.711..  Test Accuracy: 0.584
Epoch: 1/1..  Training Loss: 1.725..  Test Loss: 0.926..  Test Accuracy: 0.747
Epoch: 1/1..  Training Loss: 1.258..  Test Loss: 0.650..  Test Accuracy: 0.843
Training complete.
Testing network ...
Test Loss: 0.748..  Test Accuracy: 0.818
Testing complete.
Saving checkpoint.
Checkpoint saved to: checkpoint.pth.
Loading network from checkpoint: checkpoint.pth.
Network loaded.
Testing network ...
Test Loss: 0.748..  Test Accuracy: 0.818
Testing complete.
```
