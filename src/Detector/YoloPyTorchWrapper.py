#!/usr/bin/env python

# Core libraries
import os
import sys

# DL stuff
import torch
from torch.autograd import Variable

# PyTorch-YOLOv3
sys.path.append("/home/ca0513/PyTorch-YOLOv3")
from models import *
from utils.utils import *

# My libraries
from Utilities.ImageUtils import ImageUtils

"""
NOT WORKING YET
"""

class YoloPyTorchWrapper(object):
	# Class constructor
	def __init__(self, load_weights=True):
		"""
		Class attributes
		"""

		# Which fold are we wanting to load up?
		fold = 0

		# Confidence threshold for detections
		self.__conf_thresh = 0.5

		# NMS threshold
		self.__nms_thresh = 0.5

		# What is the base directory for the PyTorch-YOLOv3 source code
		self.__base_dir = "/home/ca0513/PyTorch-YOLOv3"

		# Where is the model definition we should use
		self.__model_path = os.path.join(self.__base_dir, "config/yolov3-custom.cfg")
		assert os.path.exists(self.__model_path)

		# Where the weights are
		self.__weights_path = os.path.join(self.__base_dir, f"output/fold-{fold}_best.pth")
		assert os.path.exists(self.__weights_path)

		# What device to put stuff on
		self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Use the GPU if we can
		self.__tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

		"""
		Class objects
		"""

		# Define the darknet model
		self.__model = Darknet(self.__model_path).to(self.__device)

		"""
		Class setup
		"""

		# If we're supposed to load model weights
		if load_weights:
			# Load weights from the defined path
			self.__model.load_state_dict(torch.load(self.__weights_path))

			# Put the model in evaluation mode
			self.__model.eval()

	"""
	Public methods
	"""

	def detect(self, image):
		# Convert the image into a tensor in the correct PyTorch format
		image = ImageUtils.npToTorch([image])[0]

		# Insert this image into a batch with size 1
		batch = image[None, :, :, :]

		# Convert tensor to PyTorch variable
		batch = Variable(batch.type(self.__tensor), requires_grad=False)

		with torch.no_grad():
			# Get the output from the network
			outputs = self.__model(batch)

			# Apply NMS on the outputs
			outputs = non_max_suppression(outputs, conf_thres=self.__conf_thresh, nms_thres=self.__nms_thresh)

		return outputs

	"""
	(Effectively) private methods
	"""

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
	pass
	