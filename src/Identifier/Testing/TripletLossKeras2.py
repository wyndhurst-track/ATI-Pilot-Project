#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("/home/will/work/1-RA/src")
from tqdm import tqdm

# Keras libraries
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import random
import matplotlib.patheffects as PathEffects
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import pickle
import matplotlib.pyplot as plt
from itertools import permutations
import seaborn as sns
from keras.datasets import mnist
from sklearn.manifold import TSNE
from sklearn.svm import SVC

# My libraries
from config import cfg
from Datasets.CowIdentityDatasets import CowIdentityDatasets

"""
Class for learning a cow embedding using triplet loss in keras

Code is mainly from: https://github.com/KinWaiCheuk/Triplet-net-keras
"""

class TripletLossKeras(object):
	# Class constructor
	def __init__(self):
		"""
		Class attributes
		"""

		# Where to save models
		self.__save_dir = os.path.join(cfg.ID.DIR, "data/models/triplet_model.hdf5")

		# Where to find the datasets
		self.__train_data_dir = os.path.join(cfg.ID.DIR, "data/cows_train")
		self.__test_data_dir = os.path.join(cfg.ID.DIR, "data/cows_test")

		"""
		Class objects
		"""

		"""
		Class setup
		"""

		# Make sure GPU(s) are available
		assert len(K.tensorflow_backend._get_available_gpus()) >= 1

		self.__setup()

	"""
	Public methods
	"""

	def trainNetwork(self):
		pass

	def testNetwork(self):
		pass

	def testMnist(self):
		# Get the data
		# (x_train, y_train), (x_test, y_test) = mnist.load_data()
		dataset = CowIdentityDatasets("CowID-PhD")
		(x_train, y_train), (x_test, y_test) = dataset.getData()

		# Size of the embedding layer
		embedding_size = 256

		# Number of epochs to train for
		epochs = 1000

		# Batch size
		batch_size = 32

		# Get the image size
		image_w = x_train.shape[2]
		image_h = x_train.shape[1]

		single_chan = False
		try:
			image_c = x_train.shape[3]
		except IndexError:
			single_chan = True

		# Whether we're training or testing
		train = True

		# Define the input layers
		anchor_input = Input((image_w, image_h, 1, ), name='anchor_input')
		positive_input = Input((image_w, image_h, 1, ), name='positive_input')
		negative_input = Input((image_w, image_h, 1, ), name='negative_input')

		# Shared embedding layer for positive and negative items
		Shared_DNN = self.__create_base_network([image_w, image_h, 1, ], embedding_size)

		# Define the three networks sharing weights
		encoded_anchor = Shared_DNN(anchor_input)
		encoded_positive = Shared_DNN(positive_input)
		encoded_negative = Shared_DNN(negative_input)

		if train:
			# Flatten it from wxh images to a w*h*c-long vector
			if single_chan:
				x_train_flat = x_train.reshape(-1,image_w*image_h)
				x_test_flat = x_test.reshape(-1,image_w*image_h)
			else:
				x_train_flat = x_train.reshape(-1,image_w*image_h*image_c)
				x_test_flat = x_test.reshape(-1,image_w*image_h*image_c)

			# # Visualise the mnist training and validation set in TSNE
			# tsne = TSNE()
			# train_tsne_embeds = tsne.fit_transform(x_train_flat[:512])
			# self.__scatter(train_tsne_embeds, y_train[:512], "Samples from Training Data")

			# eval_tsne_embeds = tsne.fit_transform(x_test_flat[:512])
			# self.__scatter(eval_tsne_embeds, y_test[:512], "Samples from Validation Data")

			# Generate our triplets
			X_train, X_test = self.__generate_triplet(	x_train_flat,
														y_train, 
														ap_pairs=150, 
														an_pairs=150,
														testsize=0.2	)

			# Optimiser
			adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

			# Concatenate the outputs of each network
			merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

			# Compile the model
			model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
			model.compile(loss=self.__batch_hard_triplet_loss_tf, optimizer=adam_optim)

			model.summary()

			Anchor = X_train[:,0,:].reshape(-1,image_w,image_h,1)
			Positive = X_train[:,1,:].reshape(-1,image_w,image_h,1)
			Negative = X_train[:,2,:].reshape(-1,image_w,image_h,1)
			Anchor_test = X_test[:,0,:].reshape(-1,image_w,image_h,1)
			Positive_test = X_test[:,1,:].reshape(-1,image_w,image_h,1)
			Negative_test = X_test[:,2,:].reshape(-1,image_w,image_h,1)

			Y_dummy = np.empty((Anchor.shape[0],300))
			Y_dummy2 = np.empty((Anchor_test.shape[0],1))

			# Train!
			model.fit(	[Anchor,Positive,Negative],
						y=Y_dummy,
						validation_data=([Anchor_test,Positive_test,Negative_test],Y_dummy2), 
						batch_size=batch_size, 
						epochs=epochs	)

			self.__saveModel()

		# Just test/visualise
		else:
			# Load the model again
			trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)
			trained_model.load_weights(self.__save_dir)

			# Visualise the learned embedding
			tsne = TSNE()
			X_train_trm = trained_model.predict(x_train[:batch_size].reshape(-1,image_w,image_h,1))
			X_test_trm = trained_model.predict(x_test[:batch_size].reshape(-1,image_w,image_h,1))
			train_tsne_embeds = tsne.fit_transform(X_train_trm)
			eval_tsne_embeds = tsne.fit_transform(X_test_trm)

			self.__scatter(train_tsne_embeds, y_train[:batch_size], "Training Data After TNN")
			self.__scatter(eval_tsne_embeds, y_test[:batch_size], "Validation Data After TNN")

			# Train a simple classifier on the embedding space
			# X_train_trm = trained_model.predict(x_train.reshape(-1,28,28,1))
			# X_test_trm = trained_model.predict(x_test.reshape(-1,28,28,1))

			# Classifier_input = Input((4,))
			# Classifier_output = Dense(10, activation='softmax')(Classifier_input)
			# Classifier_model = Model(Classifier_input, Classifier_output)

			# Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

			# Classifier_model.fit(	X_train_trm,
			# 						y_train_onehot, 
			# 						validation_data=(X_test_trm,y_test_onehot),
			# 						epochs=10	)

	"""
	(Effectively) private methods
	"""

	# Setup the class for operation
	def __setup(self):
		# Get the ResNet50 model with weights pre-trained on imagenet. Include top dictates
		# whether the last fully connected layer is included
		# self.__model = ResNet50(include_top=False, weights="imagenet")
		pass

	def __create_base_network(self, in_dims, embedding_size):
		"""
		Base network to be shared.
		"""
		model = Sequential()
		model.add(Conv2D(128,(7,7),padding='same',input_shape=(in_dims[0],in_dims[1],in_dims[2],),activation='relu',name='conv1'))
		model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool1'))
		model.add(Conv2D(256,(5,5),padding='same',activation='relu',name='conv2'))
		model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool2'))
		model.add(Flatten(name='flatten'))
		model.add(Dense(embedding_size, name='embeddings'))
		# model.add(Dense(600))
		
		return model

	def __generate_triplet(self, x, y, testsize=0.3, ap_pairs=10, an_pairs=10):
		data_xy = tuple([x,y])

		trainsize = 1-testsize

		triplet_train_pairs = []
		triplet_test_pairs = []

		data_xy_set = set(data_xy[1])
		pbar = tqdm(total=len(data_xy_set))
		for data_class in sorted(data_xy_set):

			same_class_idx = np.where((data_xy[1] == data_class))[0]
			diff_class_idx = np.where(data_xy[1] != data_class)[0]
			A_P_pairs = random.sample(list(permutations(same_class_idx,2)),k=ap_pairs) #Generating Anchor-Positive pairs
			Neg_idx = random.sample(list(diff_class_idx),k=an_pairs)
			
			#train
			A_P_len = len(A_P_pairs)
			Neg_len = len(Neg_idx)
			for ap in A_P_pairs[:int(A_P_len*trainsize)]:
				Anchor = data_xy[0][ap[0]]
				Positive = data_xy[0][ap[1]]
				for n in Neg_idx:
					Negative = data_xy[0][n]
					triplet_train_pairs.append([Anchor,Positive,Negative])               
			#test
			for ap in A_P_pairs[int(A_P_len*trainsize):]:
				Anchor = data_xy[0][ap[0]]
				Positive = data_xy[0][ap[1]]
				for n in Neg_idx:
					Negative = data_xy[0][n]
					triplet_test_pairs.append([Anchor,Positive,Negative])

			pbar.update()
		pbar.close()
					
		return np.array(triplet_train_pairs), np.array(triplet_test_pairs)

	# Naive, simple triplet loss (triplets will have been mined offline)
	def __triplet_loss(self, y_true, y_pred, margin=0.4):
		"""
		Implementation of the triplet loss function
		Arguments:
		y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
		y_pred -- python list containing three objects:
				anchor -- the embedding for the anchor data
				positive -- the embedding for the positive data (similar to anchor)
				negative -- the embedding for the negative data (different from anchor)
		Returns:
		loss -- real number, value of the loss
		"""
		print('y_pred.shape = ',y_pred)
		
		total_length = y_pred.shape.as_list()[-1]
		#     print('total_lenght=',  total_lenght)
		#     total_lenght =12
		
		anchor = y_pred[:,0:int(total_length*1/3)]
		positive = y_pred[:,int(total_length*1/3):int(total_length*2/3)]
		negative = y_pred[:,int(total_length*2/3):int(total_length*3/3)]

		# distance between the anchor and the positive
		pos_dist = K.sum(K.square(anchor - positive), axis=1)

		# distance between the anchor and the negative
		neg_dist = K.sum(K.square(anchor - negative), axis=1)

		# compute loss
		basic_loss = pos_dist - neg_dist+ margin
		loss = K.maximum(basic_loss, 0.0)
	 
		return loss

	# Triplet loss implementation with online triplet generation using batch hard
	# https://omoindrot.github.io/triplet-loss
	def __batch_hard_triplet_loss_tf(self, y_true, y_pred, margin=0.4):
		# Extract the embeddings for the anchor, positive, negative
		# total_length = y_pred.shape.as_list()[-1]
		# anchor = y_pred[:,0:int(total_length*1/3)]
		# positive = y_pred[:,int(total_length*1/3):int(total_length*2/3)]
		# negative = y_pred[:,int(total_length*2/3):int(total_length*3/3)]

		# Call the tensorflow implementation
		return tf.contrib.losses.metric_learning.triplet_semihard_loss(	y_true,
																		y_pred,
																		margin=margin)

	# Define our own plot function
	def __scatter(self, x, labels, subtitle=None):
		# We choose a color palette with seaborn.
		palette = np.array(sns.color_palette("hls", 10))

		# We create a scatter plot.
		f = plt.figure(figsize=(8, 8))
		ax = plt.subplot(aspect='equal')
		sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
						c=palette[labels.astype(np.int)])
		plt.xlim(-25, 25)
		plt.ylim(-25, 25)
		ax.axis('off')
		ax.axis('tight')

		# We add the labels for each digit.
		txts = []
		for i in range(10):
			# Position of each label.
			xtext, ytext = np.median(x[labels == i, :], axis=0)
			txt = ax.text(xtext, ytext, str(i), fontsize=24)
			txt.set_path_effects([
				PathEffects.Stroke(linewidth=5, foreground="w"),
				PathEffects.Normal()])
			txts.append(txt)
			
		if subtitle != None:
			plt.suptitle(subtitle)
			
		plt.savefig(subtitle)

	def __setupData(self):
		pass

	def __loadModel(self):
		assert self.__save_dir.endswith(".hdf5")
		self.__model = load_model(self.__save_dir)
		print(f"Loaded model from: {self.__save_dir}")

	# Save the model to file
	def __saveModel(self):
		assert self.__save_dir.endswith(".hdf5")
		self.__model.save(self.__save_dir)
		print(f"Saved model to: {self.__save_dir}")

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
	identifier = TripletLossKeras()

	# Train the network up
	identifier.testMnist()

	# Test the network
	# identifier.testNetwork()