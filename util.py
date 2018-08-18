import torch
from torchvision import datasets, transforms

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'


def get_data_sets_loaders(data_dir='flowers'):
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	normal_means = [0.485, 0.456, 0.406]
	normal_std_dev = [0.229, 0.224, 0.225]

	# Transforms:
	#  - All: Resize and crop to 224x224
	#  - All: Apply normalization via mean & std dev
	#  - Train Only: Apply random scaling, cropping & flipping

	# Define your transforms for the training, validation, and testing sets
	data_transforms = {TRAIN: transforms.Compose([transforms.RandomRotation(30),
												  transforms.RandomResizedCrop(224),
												  transforms.RandomHorizontalFlip(),
												  transforms.ToTensor(),
												  transforms.Normalize(normal_means, normal_std_dev)]),
					   VALID: transforms.Compose([transforms.Resize(256),
												  transforms.CenterCrop(224),
												  transforms.ToTensor(),
												  transforms.Normalize(normal_means, normal_std_dev)]),
					   TEST: transforms.Compose([transforms.Resize(256),
												 transforms.CenterCrop(224),
												 transforms.ToTensor(),
												 transforms.Normalize(normal_means, normal_std_dev)])}

	# Load the datasets with ImageFolder
	image_datasets = {TRAIN: datasets.ImageFolder(train_dir, transform=data_transforms[TRAIN]),
					  VALID: datasets.ImageFolder(valid_dir, transform=data_transforms[VALID]),
					  TEST: datasets.ImageFolder(test_dir, transform=data_transforms[TEST])}

	# Using the image datasets and the trainforms, define the dataloaders
	dataloaders = {TRAIN: torch.utils.data.DataLoader(image_datasets[TRAIN], batch_size=64, shuffle=True),
				   VALID: torch.utils.data.DataLoader(image_datasets[VALID], batch_size=32),
				   TEST: torch.utils.data.DataLoader(image_datasets[TEST], batch_size=32)}

	return image_datasets, dataloaders
