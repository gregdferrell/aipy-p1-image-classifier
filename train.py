# Script to train a neural network on the given data directory. Use command-line options to customize the
# hyperparameters of the neural network.
#
# Prints out training loss, validation loss, and validation accuracy as the network trains. Then saves a checkpoint.
#
# To see command-line options, exec: python train.py --help
#

import argparse

from network import Network, NetworkArchitectures
from util import TEST, TRAIN, VALID, get_data_sets_loaders


def get_input_args():
	"""
	Retrieves and parses the command line arguments created and defined using the argparse module.
	:return: parse_args() -data structure that stores the command line arguments object
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("data_dir", help="the directory where the image data is stored", default="images")
	parser.add_argument("-sd", "--save_path", help="the file path to save the training checkpoint",
						default="checkpoint.pth")
	parser.add_argument("-a", "--arch", help="the neural network architecture to use",
						choices=[e.value for e in NetworkArchitectures], default=NetworkArchitectures.VGG16.value)
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

	nw = Network(arch=NetworkArchitectures(args.arch),
				 learning_rate=args.learning_rate,
				 dropout_rate=args.dropout_rate,
				 hidden_units=(args.hidden_units,))

	image_datasets, dataloaders = get_data_sets_loaders(args.data_dir)

	# Train network
	nw.train_network(epochs=args.epochs,
					 dataloader_train=dataloaders[TRAIN],
					 dataloader_valid=dataloaders[VALID],
					 class_to_idx=image_datasets[TRAIN].class_to_idx,
					 gpu=args.gpu)

	# Test network
	nw.test_network(dataloaders[TEST], args.gpu)

	# Save the checkpoint for this network
	nw.save_checkpoint(args.save_path)


if __name__ == "__main__":
	main()
