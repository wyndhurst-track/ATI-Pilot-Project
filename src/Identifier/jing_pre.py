# Core libraries
import os
import sys
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

sys.path.append("../")
import torch
from torch.autograd import Variable

# My libraries
from config import cfg
from Identifier.MetricLearning.models.embeddings import resnet50
from Utilities.ImageUtils import ImageUtils

class MetricLearningWrapperppp():
	"""
	This class wraps up the metric learning identifcation method for use in the full pipeline
	"""
	def __init__(self):
		""" Class constructor """

		# Base directory to find weights, embeddings, etc.
		self.__base_dir = cfg.ID.ML_WEIGHTS_DIR
		# The maximum input image size
		self.__img_size = cfg.ID.IMG_SIZE
		# The size of the batch when processing multiple images
		self.__batch_size = cfg.ID.BATCH_SIZE
		# The full path to find the weights for the network
		self.__weights_path = os.path.join(self.__base_dir, cfg.ID.ML_WEIGHTS_FILE)
		assert os.path.exists(self.__weights_path)

		# Create the model and load weights at the specified path
		self.__model = resnet50(ckpt_path=self.__weights_path, softmax_enabled=True, num_classes=186)
		# Put the model on the GPU and in evaluation mode
		self.__model.cuda()
		self.__model.eval()

		# Path to find the embeddings used as the "training" set in k-NN
		self.__embeddings_path0 = os.path.join(self.__base_dir, cfg.ID.ML_EMBED_0)
		self.__embeddings_path1 = os.path.join(self.__base_dir, cfg.ID.ML_EMBED_1)
		assert os.path.exists(self.__embeddings_path0)
		assert os.path.exists(self.__embeddings_path1)

		# Load the embeddings into memory
		file0 = np.load(self.__embeddings_path0)
		file1 = np.load(self.__embeddings_path1)

		# Extract the embeddings and corresponding labels
		embeddings = np.concatenate((file0['embeddings'][1:], file1['embeddings'][1:]))
		labels = np.concatenate((file0['labels'][1:], file1['labels'][1:]))

		# Create the KNN-based classifier
		self.__classifier = KNeighborsClassifier(n_neighbors=cfg.ID.K, n_jobs=-4)

		# Fit these values to the classifier
		self.__classifier.fit(embeddings, labels)
		self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# Device for tensors

	"""
	Public methods
	"""
	def predictBatch(self, images, visualise=False):
		""" Predict on a batch (list) of images """
		# We don't need the autograd engine
		with torch.no_grad():
			# Rezie and transform the images into PyTorch form
			images = ImageUtils.npToTorch(images, return_tensor=True, resize=self.__img_size)
			# Put the images on the GPU and express them as a PyTorch variable
			images = images.to(self.__device)
			# Get the predictions on this batch
			outputs = self.__model(images)
			# Express the embeddings in numpy form
			embeddings = outputs.data
			embeddings = embeddings.detach().cpu().numpy()
			# Classify the output embeddings using k-NN
			predictions = self.__classifier.predict(embeddings).astype(int)

			if visualise:
				# Convert the images back to numpy
				np_images = images.detach().cpu().numpy().astype(np.uint8)

				# Iterate through each image and display it
				for i in range(np_images.shape[0]):
					# Extract the image
					disp_img = np_images[i,:,:,:]
					# Transpose it back to HWC from CWH
					disp_img = disp_img.transpose(1, 2, 0)
					cv2.imshow(f"Prediction = {predictions[i]}", disp_img)
					print(f"Prediction = {predictions[i]}")
					cv2.waitKey(0)

		return predictions

# Entry method/unit testing method
if __name__ == '__main__':     ##### WORKS WELL
	# Create an instance and test on a batch of images
	identifier = MetricLearningWrapperppp()
	base_dir = "/home/will/work/1-RA/src/Datasets/data/RGBDCows2020/Identification/RGB"
	img0 = cv2.imread(os.path.join(base_dir, "114/2020-02-26_12-51-59_image_roi_001.jpg")) # Class 114
	img1 = cv2.imread(os.path.join(base_dir, "096/image_0000521_2020-02-10_13-16-47_roi_001.jpg")) # Class 96
	identifier.predictBatch([img0, img1], visualise=0)
