from collections import OrderedDict
from enum import Enum
import torch
from torch import nn, optim
from torchvision import models
from typing import Dict, Tuple


class NetworkArchitectures(Enum):
	"""
	Enum representing the network architectures available to this module.
	"""
	VGG11 = 'vgg11'
	VGG13 = 'vgg13'
	VGG16 = 'vgg16'
	VGG19 = 'vgg19'


class Network:
	def __init__(self, arch: NetworkArchitectures = NetworkArchitectures.VGG16, learning_rate: float = 0.0001,
				 dropout_rate: float = 0.2, input_size: int = 25088, hidden_units: Tuple = (12544,),
				 output_size: int = 102, model_state_dict: Dict = None, epochs: int = 0, class_to_idx: Dict = None):
		self.learning_rate = learning_rate
		self.dropout_rate = dropout_rate
		self.input_size = input_size
		self.hidden_units = hidden_units
		self.output_size = output_size
		self.epochs = epochs
		self.class_to_idx = class_to_idx

		# Build model using transfer learning based off of input arch
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

		self.criterion = nn.NLLLoss()
		self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)

		if model_state_dict:
			self.model_state_dict = model_state_dict
			self.model.load_state_dict(model_state_dict)

	def create_classifier(self):
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

	def validate_network(self, dataloader_valid, gpu=False):
		# Setup Cuda
		device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
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

	def train_network(self, epochs: int, dataloader_valid, dataloader_train, class_to_idx, gpu=False):
		# Setup Cuda
		device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
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
