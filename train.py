# Script to train a neural network on the given data directory. Prints out training loss, validation loss, and
# validation accuracy as the network trains.
#
# Basic usage: python train.py data_dir
#
# Options:
# - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# - Choose architecture: python train.py data_dir --arch "vgg13"
# - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# - Use GPU for training: python train.py data_dir --gpu

import argparse

from network import Network, NetworkArchitectures
from util import get_data_sets_loaders, TEST, TRAIN, VALID


def get_input_args():
	"""
	Retrieves and parses the command line arguments created and defined using the argparse module.
	:return: parse_args() -data structure that stores the command line arguments object
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("data_dir", help="the directory where the image data is stored", default="images")
	parser.add_argument("-sd", "--save_dir", help="the directory to save the training checkpoint", default=".")
	parser.add_argument("-a", "--arch", help="the neural network architecture to use",
						choices=[NetworkArchitectures.VGG11, NetworkArchitectures.VGG13, NetworkArchitectures.VGG16,
								 NetworkArchitectures.VGG19], default=NetworkArchitectures.VGG16)
	parser.add_argument("-lr", "--learning_rate", help="the learning rate of the network when training", type=float,
						default=.0001)
	parser.add_argument("-dr", "--dropout_rate", help="the dropout rate of the network when training", type=float,
						default=.2)
	parser.add_argument("-hu", "--hidden_units", help="the number of hidden units in the network", default=12544)
	parser.add_argument("-e", "--epochs", help="the number of epochs to use when training", type=int, default=6)
	parser.add_argument("-g", "--gpu", help="flag to use a GPU when training", action="store_true")
	parser.add_argument("-v", "--verbose", help="flag to print verbose output", action="store_true")

	args = parser.parse_args()

	if args.verbose:
		print(f"Input args: {args}")

	return args


def main():
	args = get_input_args()

	nw = Network(arch=args.arch,
				 learning_rate=args.learning_rate,
				 dropout_rate=args.dropout_rate,
				 hidden_units=(args.hidden_units,))

	image_datasets, dataloaders = get_data_sets_loaders(args.data_dir)

	print(type(image_datasets))
	print(type(dataloaders))

	nw.train_network(epochs=args.epochs,
					 dataloader_train=dataloaders[TRAIN],
					 dataloader_valid=dataloaders[VALID],
					 class_to_idx=image_datasets[TRAIN].class_to_idx,
					 gpu=args.gpu)


if __name__ == "__main__":
	main()
