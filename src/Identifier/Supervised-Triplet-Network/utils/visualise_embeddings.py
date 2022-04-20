import os
import sys
import cv2
import pickle
import json
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patheffects as PathEffects
from tqdm import tqdm

# Home windows machine
if os.path.isdir("D:\\Work"): sys.path.append("D:\\Work\\ATI-Pilot-Project\\src")

# My libraries
from Utilities.DataUtils import DataUtils
from Utilities.ImageUtils import ImageUtils

# Data paths
# all_embedding_train_path = "../train_triplet_cnn_cow_id_temp_x1_full.npz"
# all_embedding_test_path = "../test_triplet_cnn_cow_id_temp_x1_full.npz"
# known_embedding_train_path = "../train_triplet_cnn_cow_id_temp_x1_known.npz"
# known_embedding_test_path = "../test_triplet_cnn_cow_id_temp_x1_known.npz"
# novel_embedding_train_path = "../train_triplet_cnn_cow_id_temp_x1_novel.npz"
# novel_embedding_test_path = "../test_triplet_cnn_cow_id_temp_x1_novel.npz"

#base_dir = "/home/ca0513/CEiA/results/SoftmaxRTL/50-50/fold_0"
base_dir = "D:\\Work\\results\\CEiA\\SRTL\\05\\rep_0"
# base_dir = "/home/will/work/CEiA/results/STL/90-10/fold_0"
# base_dir = "/home/will/work/CEiA/results/STL/50-50/fold_0"
# base_dir = "/home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network/results/fold_0"

# all_embedding_train_path = os.path.join(base_dir, "train_triplet_cnn_open_cows_temp_x1_full.npz")
# all_embedding_test_path = os.path.join(base_dir, "test_triplet_cnn_open_cows_temp_x1_full.npz")
# known_embedding_train_path = os.path.join(base_dir, "train_triplet_cnn_open_cows_temp_x1_known.npz")
# known_embedding_test_path = os.path.join(base_dir, "test_triplet_cnn_open_cows_temp_x1_known.npz")
# novel_embedding_train_path = os.path.join(base_dir, "train_triplet_cnn_open_cows_temp_x1_novel.npz")
# novel_embedding_test_path = os.path.join(base_dir, "test_triplet_cnn_open_cows_temp_x1_novel.npz")

train_embeddings_path = os.path.join(base_dir, "train_embeddings.npz")
valid_embeddings_path = os.path.join(base_dir, "valid_embeddings.npz")
test_embeddings_path = os.path.join(base_dir, "test_embeddings.npz")

# Number of classes
num_classes = 46

# Define our own plot function
def scatter(x, labels, subtitle=None, overlay_class_examples=False, class_examples=None, enable_labels=True):
	# Load the dictionary of folds (which classes are unknown)
	curr_rep = 0
	unknown_ratio = 0.5
	# folds_fp = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/2-folds.pkl"
	# folds_fp = "D:\\Work\\ATI-Pilot-Project\\src\\Datasets\\data\\OpenSetCows2019\\2-folds.pkl"
	folds_fp = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\OpenCows2020\\identification\\images\\known_unknown_splits.json"

	# We choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", num_classes+1))

	# Make a marker array based on which are the known and unknown classes currently
	known = np.ones(labels.shape)

	if os.path.exists(folds_fp):
		with open(folds_fp) as handle:
			unknowns = json.load(handle)

		# Mark which classes are novel/unseen
		for i in range(labels.shape[0]):
			if str(int(labels[i])).zfill(3) in unknowns[str(unknown_ratio)][curr_rep]['unknown']: 
				known[i] = 0
	else: folds_dict = {}

	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	# KNOWN
	sc0 = ax.scatter(	x[known==1,0], x[known==1,1], 
						lw=0, s=40, 
						c=palette[labels[known==1].astype(np.int)], 
						marker="o",
						label="known"	)
	# UNKNOWN
	sc1 = ax.scatter(	x[known==0,0], x[known==0,1],
						lw=0, s=40,
						c=palette[labels[known==0].astype(np.int)],
						marker="^",
						label="unknown"	)
	# plt.legend(loc="lower right")
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')

	# We add the labels for each digit.
	if enable_labels:
		for i in range(1,num_classes):
			if os.path.exists(folds_fp):
				# Colour the text based on whether its a known or unknown class
				if str(i).zfill(3) in unknowns[str(unknown_ratio)][curr_rep]['unknown']: text_color = "red"
				else: text_color = "black"
			else: text_color = "black"

			# Position of each label.
			xtext, ytext = np.median(x[labels == i, :], axis=0)
			txt = ax.text(xtext, ytext, str(i), fontsize=24, color=text_color)
			txt.set_path_effects([
				PathEffects.Stroke(linewidth=5, foreground="w"),
				PathEffects.Normal()])
		
	# plt.show()
	plt.tight_layout()
	plt.savefig(subtitle)

def plotEmbeddings():
	# Load them into memory
	train_embeddings = np.load(train_embeddings_path)
	valid_embeddings = np.load(valid_embeddings_path)
	test_embeddings = np.load(test_embeddings_path)

	print("Loaded embeddings")

	# Visualise the learned embedding via TSNE
	visualiser = TSNE(n_components=2, perplexity=30)
	# visualiser = PCA(n_components=2)

	# Perform TSNE magic
	tsne_train = visualiser.fit_transform(train_embeddings['embeddings'])
	tsne_valid = visualiser.fit_transform(valid_embeddings['embeddings'])
	tsne_test = visualiser.fit_transform(test_embeddings['embeddings'])

	print("Visualisation computed")

	# Plot the results and save to file
	scatter(tsne_train, train_embeddings['labels'], f"train_embeddings.pdf")
	scatter(tsne_valid, valid_embeddings['labels'], f"valid_embeddings.pdf")
	scatter(tsne_test, test_embeddings['labels'], f"test_embeddings.pdf")

# Plots the embeddings for a particular openness / fold for all different loss functions to evaluate the contrast in
# clusterings
def plotEmbeddingsComparison():
	# Parameters
	openness = "05"			# How open the problem should be
	unknown_ratio = "0.5"	# How open the problem should be
	repitition = 0				# Which repitition to render
	train = 1				# Whether to render the train or test set embeddings
	use_set = 0				# Which set to use (full, known, novel)
	display_class_labels = False # Overlay class labels on the embedding centroids

	# The list of sets
	data_sets = ["full", "known", "novel"]

	# Filenames
	train_file = f"train_embeddings.npz"
	test_file = f"test_embeddings.npz"

	# On home windows machine
	base_dir = "D:\\Work\\results\\CEiA"

	# Generate the embeddings directories
	STL_dir = os.path.join(base_dir, "TL", openness, f"rep_{repitition}")
	RTL_dir = os.path.join(base_dir, "RTL", openness, f"rep_{repitition}")
	SoftmaxTL_dir = os.path.join(base_dir, "STL", openness, f"rep_{repitition}")
	SoftmaxRTL_dir = os.path.join(base_dir, "SRTL", openness, f"rep_{repitition}")

	print("Loading the embeddings")

	# Load the embeddings
	STL = {0: np.load(os.path.join(STL_dir, test_file)), 1: np.load(os.path.join(STL_dir, train_file))}
	RTL = {0: np.load(os.path.join(RTL_dir, test_file)), 1: np.load(os.path.join(RTL_dir, train_file))}
	SoftmaxTL = {0: np.load(os.path.join(SoftmaxTL_dir, test_file)), 1: np.load(os.path.join(SoftmaxTL_dir, train_file))}
	SoftmaxRTL = {0: np.load(os.path.join(SoftmaxRTL_dir, test_file)), 1: np.load(os.path.join(SoftmaxRTL_dir, train_file))}

	# Visualise using TSNE
	visualiser = TSNE(n_components=2)

	print("Performing TSNE")

	# Perform TSNE magic
	pbar = tqdm(total=4)
	STL_TSNE = visualiser.fit_transform(STL[train]['embeddings']); pbar.update()
	RTL_TSNE = visualiser.fit_transform(RTL[train]['embeddings']); pbar.update()
	SoftmaxTL_TSNE = visualiser.fit_transform(SoftmaxTL[train]['embeddings']); pbar.update()
	SoftmaxRTL_TSNE = visualiser.fit_transform(SoftmaxRTL[train]['embeddings']); pbar.update()
	pbar.close()

	print("Rendering the embeddings")

	# Actually render the embeddings
	pbar = tqdm(total=4)
	scatter(STL_TSNE, STL[train]['labels'], "TripletLoss.pdf", enable_labels=display_class_labels); pbar.update()
	scatter(RTL_TSNE, RTL[train]['labels'], "ReciprocalTripletLoss.pdf", enable_labels=display_class_labels); pbar.update()
	scatter(SoftmaxTL_TSNE, SoftmaxTL[train]['labels'], "SoftmaxTripletLoss.pdf", enable_labels=display_class_labels); pbar.update()
	scatter(SoftmaxRTL_TSNE, SoftmaxRTL[train]['labels'], "SoftmaxReciprocalTripletLoss.pdf", enable_labels=display_class_labels); pbar.update()
	pbar.close()

# Plot the embedding for some run, where an example for each class is overlaid at the centroid of its embeddings
def plotClassOverlay():
	# Parameters
	unknown_ratio = "0.5"		# How open the problem should be
	repitition = 0				# Which fold to render
	train = 1				# Whether to render the train or test set embeddings
	use_set = 0				# Which set to use (full, known, novel)

	# Filenames
	train_file = f"train_embeddings.npz"
	test_file = f"test_embeddings.npz"

	# On home windows machine
	base_dir = "D:\\Work\\results\\CEiA"

	# Where to find the splits for the dataset
	# splits_dir = "D:\\Work\\ATI-Pilot-Project\\src\\Datasets\\data\\OpenSetCows2019\\2-folds.pkl"
	splits_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\OpenCows2020\\identification\\images\\known_unknown_splits.json"
	with open(splits_dir) as handle:
		splits_dict = json.load(handle)
	splits = splits_dict[unknown_ratio][repitition]

	# Generate the embeddings directories
	embed_dir = os.path.join(base_dir, "SRTL", "05", f"rep_{repitition}")

	# Load them
	embeddings = {0: np.load(os.path.join(embed_dir, test_file)), 1: np.load(os.path.join(embed_dir, train_file))}

	# Visualise using TSNE
	visualiser = TSNE(n_components=2)
	reduction = visualiser.fit_transform(embeddings[train]['embeddings'])

	# Directory to find dataset
	dataset_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\OpenCows2020\\identification\\images"

	# Load an example for each class
	class_filepaths = DataUtils.readFolderDatasetFilepathList(dataset_dir)

	# Produce the plot
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')

	xs = []
	ys = []
	margin = 2
	zoom = 0.25

	# Plot the images
	for k, filepaths in class_filepaths.items():
		# Get the labels
		labels = embeddings[train]['labels']

		# Compute the centroid
		x, y = np.median(reduction[labels==int(k), :], axis=0)

		xs.append(x)
		ys.append(y)

		# Load a random image for this class
		image = cv2.imread(random.choice(filepaths))
		image = ImageUtils.proportionallyResizeImageToMax(image, 200, 200)
		
		# Plot the example at the centroid
		imagebox = OffsetImage(image, zoom=zoom)

		if k in splits['unknown']: ab = AnnotationBbox(imagebox, (x, y), bboxprops=dict(color='red'))
		else: ab = AnnotationBbox(imagebox, (x, y))

		ax.add_artist(ab)

	ax.axis('off')
	ax.axis('tight')
	plt.xlim(min(xs)-margin, max(xs)+margin)
	plt.ylim(min(ys)-margin, max(ys)+margin)

	# plt.show()
	# plt.tight_layout()
	plt.savefig("class-overlay.pdf")

# Entry method/unit testing method
if __name__ == '__main__':
	# plotEmbeddings()
	plotEmbeddingsComparison()
	# plotClassOverlay()
