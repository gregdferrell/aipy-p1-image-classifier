# Script to predict the flower name from an image, along with the probability of that name.
#
# To see command-line options, exec: python predict.py --help
#

import argparse
import json

from network import Network


def get_input_args():
	"""
	Retrieves and parses the command line arguments created and defined using the argparse module.
	:return: parse_args() -data structure that stores the command line arguments object
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("image_path", help="the path to the image to predict the flower class for")
	parser.add_argument("-cp", "--checkpoint_path",
						help="the path to the network checkpoint to load and use for the prediction",
						default="checkpoint.pth")
	parser.add_argument("-t", "--top_k", help="return the top k most likely classes", default=5)
	parser.add_argument("-cn", "--category_names", help="the path to the category names json file",
						default="cat_to_name.json")
	parser.add_argument("-g", "--gpu", help="flag to use a GPU when training", action="store_true")
	parser.add_argument("-v", "--verbose", help="flag to print verbose output", action="store_true")

	args = parser.parse_args()

	if args.verbose:
		print(f"Input args: {args}")

	return args


def main():
	args = get_input_args()

	# Open the given category names JSON file
	with open(args.category_names, 'r') as f:
		cat_to_name = json.load(f)

	# Loading the network from the given checkpoint
	nw = Network.load_checkpoint(args.checkpoint_path)

	# Use network to get topk predictions
	probabilities, classes = nw.predict(args.image_path, args.top_k, args.gpu)
	for image_class, probability in zip(classes, probabilities):
		print(f"Class: {cat_to_name[image_class]}, Probability: {probability}")


if __name__ == "__main__":
	main()
