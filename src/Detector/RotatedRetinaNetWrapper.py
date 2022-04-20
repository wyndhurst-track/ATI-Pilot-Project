#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import cv2
import math
import numpy as np

# My libraries
from config import cfg
from Detector.RotatedRetinaNet import models
from Detector.RotatedRetinaNet.utils.image import preprocess_image, resize_image
from Detector.RotatedRetinaNet.utils.gpu import setup_gpu
from Visualiser.VisualisationManager import VisualisationManager

class RotatedRetinaNetWrapper(object):
	"""
	This class wraps up Jing's extension to RetinaNet for rotated cattle detection
	"""

	def __init__(self):
		""" Class constuctor """

		# Where to find model weights to load
		self.__base_dir = cfg.DET.RETINA_WEIGHTS_DIR

		# Where to find model weights
		self.__weights_path = os.path.join(self.__base_dir, cfg.DET.RETINA_WEIGHTS_FILE)
		assert os.path.exists(self.__weights_path)

		# Non-maximum suppression threshold
		self.__nms_thresh = cfg.DET.NMS_THRESHOLD

		# Confidence score threshold for a positive detection
		self.__conf_thresh = cfg.DET.CONF_THRESHOLD

		# Load the model
		self.__model = models.load_model(self.__weights_path, backbone_name='resnet50')
		self.__model = models.convert_model(self.__model, nms_threshold=self.__nms_thresh)

		# Which GPU to use
		# TODO: properly distribute work across GPUs in case 0 is busy
		self.__gpu = 0

		# Setup the GPU correctly in TensorFlow
		setup_gpu(self.__gpu)

	"""
	Public methods
	"""

	def detect(self, images, visualise=False):
		""" Detect on a list of images of the same dimensions """
		
		# Get the batch size (number of input images)
		batch_size = len(images)

		# List of scale images
		scaled = []

		# Scale each image
		for i, img in enumerate(images):
			# Work out the scale for the first image
			if i == 0:
				new_img, scale = resize_image(img, min_side=256, max_side=342)
				scaled.append(new_img)
			# Use this scale thereafter
			else:
				scaled.append(cv2.resize(img, None, fx=scale, fy=scale))

		# Convert the list of scaled images into a 4D batch
		batch = np.array(scaled)

		# Pre-process the images
		batch = preprocess_image(batch)

		# Make the prediction
		boxes, scores, labels = self.__model.predict_on_batch(batch)

		# Correct the boxes for scale (except for the angle)
		boxes[:,:,:-1] *= 1/scale

		# The list of detections for each image in the batch
		detections_batch = []

		# Iterate through each output in the batch
		for i in range(batch_size):
			# Detections for this image in the batch
			detections = []

			# Iterate through each detection
			for box, score, label in zip(boxes[i], scores[i], labels[i]):
				# First check the score is sufficient
				if score >= self.__conf_thresh:
					# Create a detection dict and convert to (centre, width, height)
					detection = {}
					detection['cx'] = (box[0] + box[2])/2
					detection['cy'] = (box[1] + box[3])/2
					detection['w'] = box[2] - box[0]
					detection['h'] = box[3] - box[1]
					detection['angle'] = box[4] + math.pi
					detection['score'] = score

					# Add this to the detections for this batch
					detections.append(detection)

			# Add the list of detections for this image to the outer list
			detections_batch.append(detections)

		# Visualise for debugging
		if visualise:
			# Iterate through each image in the batch
			for i, img in enumerate(images):
				# Visualise it
				VisualisationManager.drawRotatedBbox(img, detections_batch[i], display=True)

		return detections_batch

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

	@staticmethod
	def removeOptimiserWeights(weights_dir):
		""" Remove optimiser weights from H5 file to prevent bug in Keras """

		import h5py

		with h5py.File(weights_dir, "a") as f:
			print("Keys: %s" % f.keys())
			del f['optimizer_weights']
			print("Keys: %s" % f.keys())
			print(list(f['model_weights']))

# Entry method/unit testing method
if __name__ == '__main__':
	# Create an instance and detect on a single image
	# detector = RotatedRetinaNetWrapper()
	# img = cv2.imread("D:\\Work\\Data\\RGBDCows2020\\Detection\\RGB\\00070.jpg")
	# boxes, scores, labels = detector.detect([img])
	# print(boxes)
	# print(boxes.shape, scores.shape, labels.shape)
	# print("Finished detecting")

	# Create an instance and detect on a batch of 2 images
	detector = RotatedRetinaNetWrapper()
	img0 = cv2.imread("D:\\Work\\Data\\RGBDCows2020\\Detection\\RGB\\00070.jpg") # 2 - 3 cows
	img1 = cv2.imread("D:\\Work\\Data\\RGBDCows2020\\Detection\\RGB\\00376.jpg") # 1 - 2 cows
	detections = detector.detect([img0, img1], visualise=True)
	print("Finished detecting")

	# Modify a saved weights file
	# base_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Weights\\retinaNet"
	# weights_dir = os.path.join(base_dir, "jing-70-percent-of-data_OPTIMISER_WEIGHTS_REMOVED.h5")
	# RotatedRetinaNetWrapper.removeOptimiserWeights(weights_dir)
	