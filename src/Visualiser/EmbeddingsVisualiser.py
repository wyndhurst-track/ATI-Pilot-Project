#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import pickle
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

# My libraries
from Utilities.DataUtils import DataUtils

"""
Descriptor
"""

class EmbeddingsVisualiser(object):
	# Class constructor
	def __init__(self, folder_path):
		"""
		Class attributes
		"""

		# Path to a folder full of pickle files containing the embeddings to be visualised
		self.__folder_path = folder_path

		"""
		Class objects
		"""

		# The TSNE visualiser
		self.__visualiser = TSNE(n_components=2)

		"""
		Class setup
		"""

		# Find all the pickle files in the folder
		self.__embedding_filepaths = DataUtils.allFilesAtDirWithExt(self.__folder_path, ".pkl")

	"""
	Public methods
	"""	

	def renderTSNE(self):
		print("Rendering TSNE plots")

		# Iterate over all pickle files
		for filepath in tqdm(self.__embedding_filepaths):
			# Load the dictionary
			with open(filepath, "rb") as handle:
				embeddings_dict = pickle.load(handle)

			# Prefix for saving files
			prefix = os.path.basename(filepath)[:-4]

			# Draw the various plots
			self.__renderEmbedding(embeddings_dict["known_train"][0], embeddings_dict["known_train"][1], f"{prefix}_known_train")
			self.__renderEmbedding(embeddings_dict["known_test"][0], embeddings_dict["known_test"][1], f"{prefix}_known_test")
			self.__renderEmbedding(embeddings_dict["unknown_train"][0], embeddings_dict["unknown_train"][1], f"{prefix}_unknown_train")
			self.__renderEmbedding(embeddings_dict["unknown_test"][0], embeddings_dict["unknown_test"][1], f"{prefix}_unknown_test")

	def display(self):
		pass

	"""
	(Effectively) private methods
	"""

	def __renderEmbedding(self, embedding, labels, subtitle):
		print(f"Rendering embedding: {subtitle}")

		# Fit the transform
		x = self.__visualiser.fit_transform(embedding)

		num_classes = 46

		# We choose a color palette with seaborn.
		palette = np.array(sns.color_palette("hls", num_classes+1))

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
		for i in range(num_classes):
			# Position of each label.
			xtext, ytext = np.median(x[labels == i, :], axis=0)
			txt = ax.text(xtext, ytext, str(i), fontsize=24)
			txt.set_path_effects([
				PathEffects.Stroke(linewidth=5, foreground="w"),
				PathEffects.Normal()])
			txts.append(txt)
			
		plt.suptitle(subtitle)

		plt.savefig(f"{os.path.join(self.__folder_path, subtitle)}.pdf")

		print(f"Saved visualisation to: {subtitle}")

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
	# Path to folder with embeddings to be visualised
	folder_path = "/home/will/work/1-RA/src/Identifier/EmbeddedSpace/Results/fold_1/embeddings"
	folder_path = "/home/will/Desktop/io/fold_1"
	
	# Create our visualiser object
	ev = EmbeddingsVisualiser(folder_path)

	# Draw the embeddings
	ev.renderTSNE()

	# Display them sequentially
	# ev.display()
