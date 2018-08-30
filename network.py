# Neural network class and related enums used by training & prediction scripts.
#

from collections import OrderedDict
from enum import Enum
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from util import process_image


class NetworkArchitectures(Enum):
	"""
	Enum representing the network architectures available to this module.
	"""
	VGG11 = 'vgg11'
	VGG13 = 'vgg13'
	VGG16 = 'vgg16'
	VGG19 = 'vgg19'


class CPProps(Enum):
	"""
	Enum representing the dict keys for the network checkpoint properties.
	"""
	ARCH = 'arch'
	CLASS_TO_IDX = 'class_to_idx'
	CRITERION = 'criterion'
	DROPOUT_RATE = 'dropout_rate'
	EPOCHS = 'epochs'
	HIDDEN_UNITS = 'hidden_units'
	INPUT_SIZE = 'input_size'
	LEARNING_RATE = 'learning_rate'
	MODEL_STATE_DICT = 'model_state_dict'
	OUTPUT_SIZE = 'output_size'


class Network:
	def __init__(self, arch: NetworkArchitectures = NetworkArchitectures.VGG16, learning_rate: float = 0.0001,
				 dropout_rate: float = 0.2, input_size: int = 25088, hidden_units: Tuple = (12544,),
				 output_size: int = 102, model_state_dict: Dict = None, epochs: int = 0, class_to_idx: Dict = None,
				 criterion=nn.NLLLoss()):
		"""
		Constructor for a network.
		:param arch: the network architecture
		:param learning_rate: the learning rate used when training the network
		:param dropout_rate: the dropout rate used when training the network
		:param input_size: the input size of the classifier
		:param hidden_units: the number of nodes in the classifier hidden layer
		:param output_size: the output size of the classifier (should equal the number of categories)
		:param model_state_dict: the state_dict of the model; used for saving & loading training progress
		:param epochs: the number of epochs this network has been trained
		:param class_to_idx: a dict of classes to indices (usually taken from a training image dataset)
		"""
		self.arch = arch
		self.learning_rate = learning_rate
		self.dropout_rate = dropout_rate
		self.input_size = input_size
		self.hidden_units = hidden_units
		self.output_size = output_size
		self.epochs = epochs
		self.class_to_idx = class_to_idx
		self.criterion = criterion

		# Build the model using transfer learning, basing it off of the specified input architecture
		if arch == NetworkArchitectures.VGG11:
			self.model = models.vgg11(pretrained=True)
		elif arch == NetworkArchitectures.VGG13:
			self.model = models.vgg13(pretrained=True)
		elif arch == NetworkArchitectures.VGG16:
			self.model = models.vgg16(pretrained=True)
		elif arch == NetworkArchitectures.VGG19:
			self.model = models.vgg19(pretrained=True)
		else:
			raise ValueError(f'Invalid Network Architecture: {arch}')

		# Freeze pre-trained parameters so we don't backpropagate through them
		for param in self.model.parameters():
			param.requires_grad = False
		self.model.classifier = self.create_classifier()

		self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)

		if model_state_dict:
			self.model_state_dict = model_state_dict
			self.model.load_state_dict(model_state_dict)

	def create_classifier(self):
		"""
		Creates a network classifier given the current properties of the network.
		:return: the network classifier
		"""
		layers = OrderedDict([
			('fcstart', nn.Linear(self.input_size, self.hidden_units[0])),
			('relustart', nn.ReLU()),
			('dropoutstart', nn.Dropout(self.dropout_rate)),
		])
		for i in range(len(self.hidden_units) - 1):
			layers['fc{}'.format(i)] = nn.Linear(self.hidden_units[i], self.hidden_units[i + 1])
			layers['relu{}'.format(i)] = nn.ReLU()
			layers['dropout{}'.format(i)] = nn.Dropout(self.dropout_rate)
		layers['output'] = nn.Linear(self.hidden_units[-1], self.output_size)
		layers['logsoftmax'] = nn.LogSoftmax(dim=1)
		classifier = nn.Sequential(layers)
		return classifier

	def validate_network(self, dataloader_valid: DataLoader, gpu: bool = False):
		"""
		Validates the network by predicting a test/validation image set and reporting test loss & accuracy.
		:param dataloader_valid: the dataloader to use for validation
		:param gpu: boolean indicating to use gpu or not
		:return: a tuple containing the [0]test loss and [1]accuracy
		"""
		# Setup Cuda
		device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
		self.model.to(device)

		# Set model to eval mode so it does not use dropout & other training features
		self.model.eval()

		test_loss = 0
		accuracy = 0

		with torch.no_grad():
			for test_images, test_labels in dataloader_valid:
				test_images, test_labels = test_images.to(device), test_labels.to(device)

				test_outputs = self.model(test_images)
				test_loss += self.criterion(test_outputs, test_labels).item()

				probabilities = torch.exp(test_outputs)
				equality = (test_labels.data == probabilities.max(dim=1)[1])
				accuracy += equality.type(torch.FloatTensor).mean()

		# Set model back to train mode
		self.model.train()

		return test_loss, accuracy

	def train_network(self, epochs: int, dataloader_valid: DataLoader, dataloader_train: DataLoader, class_to_idx: Dict,
					  gpu=False):
		"""
		Trains the network with the given parameters.
		:param epochs: the number of epochs to train
		:param dataloader_valid: the dataloader to use for validation
		:param dataloader_train: the dataloader to use for training
		:param class_to_idx: a dict of classes to indices (usually taken from a training image dataset)
		:param gpu: boolean indicating to use gpu or not
		"""
		print("Training ...")
		# Setup Cuda
		device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
		self.model.to(device)

		# Set model to train mode so it uses dropout & other features
		self.model.train()

		steps = 0
		print_every = 32

		for e in range(epochs):
			running_loss = 0
			for images, labels in dataloader_train:
				images, labels = images.to(device), labels.to(device)

				steps += 1

				# Clear the gradients because gradients are accumulated
				self.optimizer.zero_grad()

				# Forward and backward passes
				outputs = self.model.forward(images)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()

				if steps % print_every == 0:
					self.model.eval()

					test_loss, accuracy = self.validate_network(dataloader_valid, gpu)

					print("Epoch: {}/{}.. ".format(e + 1, epochs),
						  "Training Loss: {:.3f}.. ".format(running_loss / print_every),
						  "Test Loss: {:.3f}.. ".format(test_loss / len(dataloader_valid)),
						  "Test Accuracy: {:.3f}".format(accuracy / len(dataloader_valid)))

					running_loss = 0

					# Make sure training is back on
					self.model.train()

		self.epochs = self.epochs + epochs
		self.class_to_idx = class_to_idx
		print("Training complete.")

	def test_network(self, dataloader_test, gpu):
		"""
		Tests the network using the given testloader.
		:param dataloader_test: the dataloader to use for testing
		:param gpu: boolean indicating to use gpu or not
		:return: nothing -prints out test loss & accuracy
		"""
		print("Testing network ...")
		self.model.eval()
		test_loss, accuracy = self.validate_network(dataloader_test, gpu)

		print("Test Loss: {:.3f}.. ".format(test_loss / len(dataloader_test)),
			  "Test Accuracy: {:.3f}".format(accuracy / len(dataloader_test)))
		print("Testing complete.")

	def save_checkpoint(self, checkpoint_filepath):
		"""
		Saves a checkpoint to the given filepath.
		:param checkpoint_filepath: the path to the checkpoint file to save
		"""
		print("Saving checkpoint.")
		checkpoint = {
			CPProps.ARCH: self.arch,
			CPProps.CLASS_TO_IDX: self.class_to_idx,
			CPProps.DROPOUT_RATE: self.dropout_rate,
			CPProps.EPOCHS: self.epochs,
			CPProps.HIDDEN_UNITS: self.hidden_units,
			CPProps.INPUT_SIZE: self.input_size,
			CPProps.LEARNING_RATE: self.learning_rate,
			CPProps.MODEL_STATE_DICT: self.model.state_dict(),
			CPProps.OUTPUT_SIZE: self.output_size,
			CPProps.CRITERION: self.criterion
		}

		torch.save(checkpoint, checkpoint_filepath)
		print(f"Checkpoint saved to: {checkpoint_filepath}.")

	@staticmethod
	def load_checkpoint(checkpoint_filepath, gpu=False):
		"""
		Creates and returns a network from the given checkpoint filepath.
		:param checkpoint_filepath: the path to the checkpoint file to load
		:param gpu: flag to use a GPU when loading checkpoint
		:return: a new instance of a Network loaded from the given checkpoint
		"""
		print(f"Loading network from checkpoint: {checkpoint_filepath}.")
		device_map_location = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
		checkpoint = torch.load(checkpoint_filepath, map_location=device_map_location)

		network = Network(
			arch=checkpoint[CPProps.ARCH],
			class_to_idx=checkpoint[CPProps.CLASS_TO_IDX],
			dropout_rate=checkpoint[CPProps.DROPOUT_RATE],
			epochs=checkpoint[CPProps.EPOCHS],
			hidden_units=checkpoint[CPProps.HIDDEN_UNITS],
			input_size=checkpoint[CPProps.INPUT_SIZE],
			learning_rate=checkpoint[CPProps.LEARNING_RATE],
			model_state_dict=checkpoint[CPProps.MODEL_STATE_DICT],
			output_size=checkpoint[CPProps.OUTPUT_SIZE],
			criterion=checkpoint[CPProps.CRITERION]
		)

		print("Network loaded.")
		return network

	def predict(self, image_path, topk=5, gpu=False):
		"""
		Predict the class (or classes) of an image using a trained deep learning model.
		:param image_path:
		:param topk: the number of classes to return
		:param gpu: flag to use a GPU when predicting
		:return: tuple containing [0]the list of probabilities and [1]the list of classes
		"""
		# Setup Cuda
		device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
		self.model.to(device)

		# Make sure model is in eval mode
		self.model.eval()

		# Process image into numpy image, then convert to torch tensor
		np_image = process_image(image_path)
		torch_image = torch.from_numpy(np_image)
		torch_image = torch_image.to(device)

		with torch.no_grad():
			output = self.model(torch_image.unsqueeze_(0))
			probabilities = torch.exp(output)
			kprobs, kindex = probabilities.topk(topk)

		kprobs_list = kprobs[0].cpu().numpy().tolist()
		kindex_list = kindex[0].cpu().numpy().tolist()

		# For every kindex value, look up the class and return it instead of the index
		idx_to_class = {v: k for k, v in self.class_to_idx.items()}
		class_list = [idx_to_class[idx] for idx in kindex_list]

		return kprobs_list, class_list
