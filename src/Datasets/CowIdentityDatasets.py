#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("/home/will/work/1-RA/src")
import cv2
import math
import random
import numpy as np

# My libraries
from config import cfg
from Utilities.DataUtils import DataUtils

"""
Generic wrapper for cattle identity datasets of the form:

cow0 /	001.jpg,
		002.jpg,
		...
cow1 /	001.jpg,
		002.jpg,
		...
...

Where each image in the folder cowx is a cropped RoI of cowx NOT including the head
"""

class CowIdentityDatasets():
	# Class constructor
	def __init__(self, dataset_name, load_into_mem=True):
		# Keep the name of the dataset
		self.__dataset_name = dataset_name

		# Let's see if the dataset name (folder) exists in our folder of datasets
		self.__complete_dir = os.path.join(cfg.DAT.DATA_DIR, dataset_name)
		assert os.path.isdir(self.__complete_dir)

		# Load the dataset into memory if we're supposed to
		if load_into_mem: self.__loadDataset()

	"""
	Public methods
	"""

	# Let's retrieve the data in the desired format
	def getData(self, train_test_split=0.9, shuffle=True, img_size=(224,224)):
		# Create a list of images and labels
		X = []
		y = []
		for category, images in self.__data_dict.items():
			for img in images:
				# Resize the image to match the requirement, change the label to an int
				X.append(cv2.resize(img, img_size))
				y.append(int(category))
		assert len(X) == len(y)

		# Shuffle both lists deterministically (so that train/test sets are consistent
		# between executions) by providing a random seed to shuffle()
		if shuffle:
			temp = list(zip(X, y))
			random.Random(4).shuffle(temp)
			X, y = zip(*temp)

		# Find the index of the split
		split_idx = int(math.floor(len(X) * train_test_split))

		# Split stuff
		X_train = np.array(X[:split_idx])
		y_train = np.array(y[:split_idx])
		X_test = np.array(X[split_idx:])
		y_test = np.array(y[split_idx:])

		return (X_train, y_train), (X_test, y_test)

	"""
	(Effectively) private methods
	"""

	# Let's load the dataset into memory from the folder
	def __loadDataset(self):
		print(f"Loading the '{self.__dataset_name}' dataset")
		self.__data_dict = DataUtils.readFolderDataset(self.__complete_dir)
		print(f"Done.")

	"""
	Getters
	"""

	"""
	Setters
	"""

	"""
	Static methods
	"""

# Entry method/unit testing method
if __name__ == '__main__':
	# Create an instance
	dataset = CowIdentityDatasets("CowID-PhD")
	dataset.getData()
	