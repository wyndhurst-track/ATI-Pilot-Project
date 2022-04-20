#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("/home/will/work/1-RA/src")
from tqdm import tqdm

# Keras libraries
import keras
from keras.applications.resnet50 import ResNet50 # https://keras.io/applications/

# My libraries
from config import cfg
from Utilities.Utility import Utility

"""
Implements basic incremental learning by freezing the last x layers of the network and fine-
tuning on new data. Network is ResNet-101 using Keras
"""

class IncrementalLearnerKeras(object):
	# Class constructor
	def __init__(self):
		"""
		Class attributes
		"""

		# Where to save models
		self.__save_dir = os.path.join(cfg.ID.ID_DIR, "data/models/IL.net")

		# Where to find the datasets
		self.__train_data_dir = os.path.join(cfg.ID.ID_DIR, "data/cows_train")
		self.__test_data_dir = os.path.join(cfg.ID.ID_DIR, "data/cows_test")

		"""
		Class objects
		"""

		"""
		Class setup
		"""

		self.__setup()

	"""
	Public methods
	"""

	def trainNetwork(self):
		pass

	def testNetwork(self):
		pass

	"""
	(Effectively) private methods
	"""

	# Setup the class for operation
	def __setup(self):
		# Get the ResNet50 model with weights pre-trained on imagenet. Include top dictates
		# whether the last fully connected layer is included
		self.__model = ResNet50(include_top=False, weights="imagenet")

	def __setupData(self):
		pass

	def __loadModel(self):
		pass

	def __saveModel(self):
		pass

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
	identifier = IncrementalLearnerKeras()

	# Train the network up
	identifier.trainNetwork()

	# Test the network
	# identifier.testNetwork()