#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("/home/will/work/1-RA/src")
from tqdm import tqdm

# Keras libraries
import keras
from keras.applications.resnet50 import ResNet50 # https://keras.io/applications/

from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA

# My libraries
from config import cfg
from Utilities.Utility import Utility

"""
Class for learning a cow embedding using triplet loss in keras

Code is mainly from: https://github.com/AdrianUng/keras-triplet-loss-mnist
"""

class TripletLossKeras(object):
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

	def testMnist(self):
		from keras.datasets import mnist

		batch_size = 256
		epochs = 25
		train_flag = False

		embedding_size = 64

		no_of_components = 2  # for visualization -> PCA.fit_transform()

		step = 10

		# The data, split between train and test sets
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255.
		x_test /= 255.
		input_image_shape = (28, 28, 1)
		x_val = x_test[:2000, :, :]
		y_val = y_test[:2000]

		# Network training...
		if train_flag == True:
			base_network = self.__create_base_network(input_image_shape, embedding_size)

			input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
			input_labels = Input(shape=(1,), name='input_label')    # input layer for labels
			embeddings = base_network([input_images])               # output of network -> embeddings
			labels_plus_embeddings = concatenate([input_labels, embeddings])  # concatenating the labels + embeddings

			# Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
			model = Model(inputs=[input_images, input_labels],
						  outputs=labels_plus_embeddings)

			model.summary()
			plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

			# train session
			opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!

			model.compile(loss=TripletLoss.triplet_loss_adapted_from_tf,
						  optimizer=opt)

			filepath = "semiH_trip_MNIST_v13_ep{epoch:02d}_BS%d.hdf5" % batch_size
			checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=25)
			callbacks_list = [checkpoint]

			# Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
			dummy_gt_train = np.zeros((len(x_train), embedding_size + 1))
			dummy_gt_val = np.zeros((len(x_val), embedding_size + 1))

			x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], x_train.shape[1], 1))
			x_val = np.reshape(x_val, (len(x_val), x_train.shape[1], x_train.shape[1], 1))

			H = model.fit(
				x=[x_train,y_train],
				y=dummy_gt_train,
				batch_size=batch_size,
				epochs=epochs,
				validation_data=([x_val, y_val], dummy_gt_val),
				callbacks=callbacks_list)
			
			plt.figure(figsize=(8,8))
			plt.plot(H.history['loss'], label='training loss')
			plt.plot(H.history['val_loss'], label='validation loss')
			plt.legend()
			plt.title('Train/validation loss')
			plt.show()
		else:

			#####
			model = load_model('semiH_trip_MNIST_v13_ep25_BS256.hdf5',
											custom_objects={'triplet_loss_adapted_from_tf':TripletLoss.triplet_loss_adapted_from_tf})


		# Test the network
		# creating an empty network
		testing_embeddings = self.__create_base_network(input_image_shape,
												 embedding_size=embedding_size)
		x_embeddings_before_train = testing_embeddings.predict(np.reshape(x_test, (len(x_test), 28, 28, 1)))
		# Grabbing the weights from the trained network
		for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[2].layers):
			weights = layer_source.get_weights()
			layer_target.set_weights(weights)
			del weights

		x_embeddings = testing_embeddings.predict(np.reshape(x_test, (len(x_test), 28, 28, 1)))
		dict_embeddings = {}
		dict_gray = {}
		test_class_labels = np.unique(np.array(y_test))

		pca = PCA(n_components=no_of_components)
		decomposed_embeddings = pca.fit_transform(x_embeddings)
	#     x_test_reshaped = np.reshape(x_test, (len(x_test), 28 * 28))
		decomposed_gray = pca.fit_transform(x_embeddings_before_train)
		
		fig = plt.figure(figsize=(16, 8))
		for label in test_class_labels:
			decomposed_embeddings_class = decomposed_embeddings[y_test == label]
			decomposed_gray_class = decomposed_gray[y_test == label]

			plt.subplot(1,2,1)
			plt.scatter(decomposed_gray_class[::step,1], decomposed_gray_class[::step,0],label=str(label))
			plt.title('before training (embeddings)')
			plt.legend()

			plt.subplot(1,2,2)
			plt.scatter(decomposed_embeddings_class[::step, 1], decomposed_embeddings_class[::step, 0], label=str(label))
			plt.title('after @%d epochs' % epochs)
			plt.legend()

		plt.show()


	"""
	(Effectively) private methods
	"""

	# Setup the class for operation
	def __setup(self):
		# Get the ResNet50 model with weights pre-trained on imagenet. Include top dictates
		# whether the last fully connected layer is included
		# self.__model = ResNet50(include_top=False, weights="imagenet")
		pass

	def __create_base_network(self, image_input_shape, embedding_size):
		"""
		Base network to be shared (eq. to feature extraction).
		"""
		input_image = Input(shape=image_input_shape)

		x = Flatten()(input_image)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.1)(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.1)(x)
		x = Dense(embedding_size)(x)

		base_network = Model(inputs=input_image, outputs=x)
		plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
		return base_network

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

class TripletLoss():
	@staticmethod
	def triplet_loss_adapted_from_tf(y_true, y_pred):
		del y_true
		margin = 1.
		labels = y_pred[:, :1]

	 
		labels = tf.cast(labels, dtype='int32')

		embeddings = y_pred[:, 1:]

		### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
		
		# Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
		# lshape=array_ops.shape(labels)
		# assert lshape.shape == 1
		# labels = array_ops.reshape(labels, [lshape[0], 1])

		# Build pairwise squared distance matrix.
		pdist_matrix = TripletLoss.pairwise_distance(embeddings, squared=True)
		# Build pairwise binary adjacency matrix.
		adjacency = math_ops.equal(labels, array_ops.transpose(labels))
		# Invert so we can select negatives only.
		adjacency_not = math_ops.logical_not(adjacency)

		# global batch_size  
		batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

		# Compute the mask.
		pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
		mask = math_ops.logical_and(
			array_ops.tile(adjacency_not, [batch_size, 1]),
			math_ops.greater(
				pdist_matrix_tile, array_ops.reshape(
					array_ops.transpose(pdist_matrix), [-1, 1])))
		mask_final = array_ops.reshape(
			math_ops.greater(
				math_ops.reduce_sum(
					math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
				0.0), [batch_size, batch_size])
		mask_final = array_ops.transpose(mask_final)

		adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
		mask = math_ops.cast(mask, dtype=dtypes.float32)

		# negatives_outside: smallest D_an where D_an > D_ap.
		negatives_outside = array_ops.reshape(
			TripletLoss.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
		negatives_outside = array_ops.transpose(negatives_outside)

		# negatives_inside: largest D_an.
		negatives_inside = array_ops.tile(
			TripletLoss.masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
		semi_hard_negatives = array_ops.where(
			mask_final, negatives_outside, negatives_inside)

		loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

		mask_positives = math_ops.cast(
			adjacency, dtype=dtypes.float32) - array_ops.diag(
			array_ops.ones([batch_size]))

		# In lifted-struct, the authors multiply 0.5 for upper triangular
		#   in semihard, they take all positive pairs except the diagonal.
		num_positives = math_ops.reduce_sum(mask_positives)

		semi_hard_triplet_loss_distance = math_ops.truediv(
			math_ops.reduce_sum(
				math_ops.maximum(
					math_ops.multiply(loss_mat, mask_positives), 0.0)),
			num_positives,
			name='triplet_semihard_loss')
		
		### Code from Tensorflow function semi-hard triplet loss ENDS here.
		return semi_hard_triplet_loss_distance

	@staticmethod
	def pairwise_distance(feature, squared=False):
		"""Computes the pairwise distance matrix with numerical stability.

		output[i, j] = || feature[i, :] - feature[j, :] ||_2

		Args:
		  feature: 2-D Tensor of size [number of data, feature dimension].
		  squared: Boolean, whether or not to square the pairwise distances.

		Returns:
		  pairwise_distances: 2-D Tensor of size [number of data, number of data].
		"""
		pairwise_distances_squared = math_ops.add(
			math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
			math_ops.reduce_sum(
				math_ops.square(array_ops.transpose(feature)),
				axis=[0],
				keepdims=True)) - 2.0 * math_ops.matmul(feature,
														array_ops.transpose(feature))

		# Deal with numerical inaccuracies. Set small negatives to zero.
		pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
		# Get the mask where the zero distances are at.
		error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

		# Optionally take the sqrt.
		if squared:
			pairwise_distances = pairwise_distances_squared
		else:
			pairwise_distances = math_ops.sqrt(
				pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

		# Undo conditionally adding 1e-16.
		pairwise_distances = math_ops.multiply(
			pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

		num_data = array_ops.shape(feature)[0]
		# Explicitly set diagonals to zero.
		mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
			array_ops.ones([num_data]))
		pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
		return pairwise_distances

	@staticmethod
	def masked_maximum(data, mask, dim=1):
		"""Computes the axis wise maximum over chosen elements.

		Args:
		  data: 2-D float `Tensor` of size [n, m].
		  mask: 2-D Boolean `Tensor` of size [n, m].
		  dim: The dimension over which to compute the maximum.

		Returns:
		  masked_maximums: N-D `Tensor`.
			The maximized dimension is of size 1 after the operation.
		"""
		axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
		masked_maximums = math_ops.reduce_max(
			math_ops.multiply(data - axis_minimums, mask), dim,
			keepdims=True) + axis_minimums
		return masked_maximums

	@staticmethod
	def masked_minimum(data, mask, dim=1):
		"""Computes the axis wise minimum over chosen elements.

		Args:
		  data: 2-D float `Tensor` of size [n, m].
		  mask: 2-D Boolean `Tensor` of size [n, m].
		  dim: The dimension over which to compute the minimum.

		Returns:
		  masked_minimums: N-D `Tensor`.
			The minimized dimension is of size 1 after the operation.
		"""
		axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
		masked_minimums = math_ops.reduce_min(
			math_ops.multiply(data - axis_maximums, mask), dim,
			keepdims=True) + axis_maximums
		return masked_minimums

# Entry method/unit testing method
if __name__ == '__main__':
	# Create an instance
	identifier = TripletLossKeras()

	# Train the network up
	identifier.testMnist()

	# Test the network
	# identifier.testNetwork()