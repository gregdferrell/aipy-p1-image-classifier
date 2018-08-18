import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

NORMAL_MEANS = (0.485, 0.456, 0.406)
NORMAL_STD_DEVIATIONS = (0.229, 0.224, 0.225)


def get_data_sets_loaders(data_dir: str = 'images'):
	"""
	Creates and returns image datasets and data loaders from a directory of images.
	:param data_dir: the path to the directory of images
	:return: datasets and dataloaders for training, testing & validation
	"""
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	# Transforms:
	#  - All: Resize and crop to 224x224
	#  - All: Apply normalization via mean & std dev
	#  - Train Only: Apply random scaling, cropping & flipping

	# Define your transforms for the training, validation, and testing sets
	data_transforms = {TRAIN: transforms.Compose([transforms.RandomRotation(30),
												  transforms.RandomResizedCrop(224),
												  transforms.RandomHorizontalFlip(),
												  transforms.ToTensor(),
												  transforms.Normalize(NORMAL_MEANS, NORMAL_STD_DEVIATIONS)]),
					   VALID: transforms.Compose([transforms.Resize(256),
												  transforms.CenterCrop(224),
												  transforms.ToTensor(),
												  transforms.Normalize(NORMAL_MEANS, NORMAL_STD_DEVIATIONS)]),
					   TEST: transforms.Compose([transforms.Resize(256),
												 transforms.CenterCrop(224),
												 transforms.ToTensor(),
												 transforms.Normalize(NORMAL_MEANS, NORMAL_STD_DEVIATIONS)])}

	# Load the datasets with ImageFolder
	image_datasets = {TRAIN: datasets.ImageFolder(train_dir, transform=data_transforms[TRAIN]),
					  VALID: datasets.ImageFolder(valid_dir, transform=data_transforms[VALID]),
					  TEST: datasets.ImageFolder(test_dir, transform=data_transforms[TEST])}

	# Using the image datasets and the transforms, define the dataloaders
	dataloaders = {TRAIN: torch.utils.data.DataLoader(image_datasets[TRAIN], batch_size=64, shuffle=True),
				   VALID: torch.utils.data.DataLoader(image_datasets[VALID], batch_size=32),
				   TEST: torch.utils.data.DataLoader(image_datasets[TEST], batch_size=32)}

	return image_datasets, dataloaders


def process_image(image_path):
	"""
	Given a path to a file, pre-process that image in preparation for making a prediction.
	:param image_path: the path to the image file
	:return: the image represented by a flattened numpy array
	"""
	im_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(NORMAL_MEANS, NORMAL_STD_DEVIATIONS)
	])

	# Open image
	im = Image.open(image_path)

	# Transform it: creates pytorch tensor
	im_transformed_tensor = im_transforms(im)

	# Return np array
	np_image = np.array(im_transformed_tensor)
	return np_image
