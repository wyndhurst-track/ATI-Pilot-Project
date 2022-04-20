# Core libraries
import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable

# Local libraries
from utilities.utils import Utilities
from models.embeddings import resnet50

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""

def evaluateModel(args):
	"""
	For a trained model, let's evaluate its performance
	"""

	# Load the relevant datasets
	train_dataset = Utilities.selectDataset(args, "train")
	test_dataset = Utilities.selectDataset(args, "test")
	assert train_dataset.getNumClasses() == test_dataset.getNumClasses()

	# Load the validation set too if we're supposed to
	if args.split == "trainvalidtest":
		valid_dataset = Utilities.selectDataset(args, "valid")

	# Define our embeddings model
	model = resnet50(	pretrained=True, 
						num_classes=train_dataset.getNumClasses(), 
						ckpt_path=args.model_path, 
						embedding_size=args.embedding_size,
						img_type=args.img_type,
						softmax_enabled=args.softmax_enabled	)
	
	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()

	# Get the embeddings and labels of the training set and testing set
	train_embeddings, train_labels = inferEmbeddings(args, model, train_dataset, "train")
	test_embeddings, test_labels = inferEmbeddings(args, model, test_dataset, "test")

	# Is there a validation set too
	if args.split == "trainvalidtest":
		valid_embeddings, valid_labels = inferEmbeddings(args, model, valid_dataset, "valid")

		# Combine the training and validation embeddings/labels to help KNN
		train_embeddings = np.concatenate((train_embeddings, valid_embeddings))
		train_labels = np.concatenate((train_labels, valid_labels))

		# Get performance on the validation and testing sets
		valid_accuracy, valid_preds = KNNAccuracy(train_embeddings, train_labels, valid_embeddings, valid_labels)

	# Get performance on the testing set
	test_accuracy, test_preds = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)

	# Path to statistics file
	stats_path = os.path.join(args.save_path, f"{args.img_type}_testing_stats.json")

	# Is there a validation set too
	if args.split == "trainvalidtest":
		# Store statistics (to produce confusion matrices)
		stats = {	'valid_labels': valid_labels.astype(int).tolist(),
					'valid_preds': valid_preds.astype(int).tolist(),
					'test_labels': test_labels.astype(int).tolist(),
					'test_preds': test_preds.astype(int).tolist()		}

		# Write it out to the console so that subprocess can pick them up and close
		sys.stdout.write(f"Validation accuracy={str(valid_accuracy)}; Testing accuracy={str(test_accuracy)}; \n")

	elif args.split == "traintest":
		# Store statistics (to produce confusion matrices)
		stats = {	'test_labels': test_labels.astype(int).tolist(),
					'test_preds': test_preds.astype(int).tolist()		}

		# Write it out to the console so that subprocess can pick them up and close
		sys.stdout.write(f"Testing accuracy={str(test_accuracy)}\n")

	# Save them to file
	with open(stats_path, 'w') as handle:
		json.dump(stats, handle, indent=4)

	sys.stdout.flush()
	sys.exit(0)

def kGridSearch(args):
	""" Perform a grid search for k nearest neighbours

	Example command:
	python test.py --model_path="D:\Work\results\CEiA\SRTL\05\rep_0\triplet_cnn_open_cows_best_x1.pkl" --mode="gridsearch" --dataset="OpenSetCows2020" --save_path="output"

	Arguments:

	"""

	# Which fold / repitition to use
	args.repeat_num = 0

	# What is the ratio of unknown classes
	args.unknown_ratio = 0.5

	# Load the relevant datasets
	train_dataset = Utilities.selectDataset(args, "train")
	test_dataset = Utilities.selectDataset(args, "valid")
	assert train_dataset.getNumClasses() == test_dataset.getNumClasses()

	# Define our embeddings model
	model = resnet50(	pretrained=True, 
						num_classes=train_dataset.getNumClasses(), 
						ckpt_path=args.model_path, 
						embedding_size=args.embedding_size,
						img_type=args.img_type,
						softmax_enabled=args.softmax_enabled	)
	
	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()

	# Get the embeddings and labels of the training set and testing set
	train_embeddings, train_labels = inferEmbeddings(args, model, train_dataset, "train")
	test_embeddings, test_labels = inferEmbeddings(args, model, test_dataset, "valid")

	# Dict to store results
	results = {}

	# Iterate from 1 to the number of testing instances
	for k in tqdm(range(1, len(test_dataset))):
		# Classify the test set
		accuracy, _ = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=k)

		# Save the accuracy for this k
		results[k] = accuracy

		# print(f"Accuracy = {accuracy} for {k}-NN classification")

	# Save them to file
	with open(os.path.join(args.save_path, f"k_grid_search.json"), 'w') as handle:
		json.dump(results, handle, indent=4)

# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4) #weights : (default = ‘uniform’) distance

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    total = len(test_labels-1)

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)

    # How many were correct?
    correct = (predictions == test_labels).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100

    return accuracy, predictions

# Infer the embeddings for a given dataset
def inferEmbeddings(args, model, dataset, split):
	# Wrap up the dataset in a PyTorch dataset loader
	data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)

	# Embeddings/labels to be stored on the testing set
	outputs_embedding = np.zeros((1,args.embedding_size))
	labels_embedding = np.zeros((1))
	total = 0
	correct = 0

	# Iterate through the training/testing portion of the dataset and get their embeddings
	#for images, _, _, labels, _ in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
	for images, _, _, labels, _ in (data_loader):
		# We don't need the autograd engine
		with torch.no_grad():
			# Put the images on the GPU and express them as PyTorch variables
			images = Variable(images.cuda())

			# Get the embeddings of this batch of images
			outputs = model(images)

			# Express embeddings in numpy form
			embeddings = outputs.data
			embeddings = embeddings.detach().cpu().numpy()

			# Convert labels to readable numpy form
			labels = labels.view(len(labels))
			labels = labels.detach().cpu().numpy()

			# Store testing data on this batch ready to be evaluated
			outputs_embedding = np.concatenate((outputs_embedding,embeddings), axis=0)
			labels_embedding = np.concatenate((labels_embedding,labels), axis=0)
	
	# If we're supposed to be saving the embeddings and labels to file
	if args.save_embeddings:
		# Construct the save path
		save_path = os.path.join(args.save_path, f"{split}_embeddings.npz")
		
		# Save the embeddings to a numpy array
		np.savez(save_path,  embeddings=outputs_embedding, labels=labels_embedding)

	return outputs_embedding, labels_embedding

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Params')
	# Required arguments
	parser.add_argument('--model_path', nargs='?', type=str, default="/home/will/Desktop/Work/output_result/fold_0/current_model_state.pkl", ##,required=True,
						help='Path to the saved model to load weights from')
	parser.add_argument('--save_path', type=str, default="/home/will/Desktop/Work/output_result/fold_0",#required=True,
						help="Where to store the embeddings and statistics")
	parser.add_argument('--split', type=str, default="trainvalidtest",
						help="Which evaluation mode to use: [trainvalidtest, traintest]")
	parser.add_argument('--mode', type=str, default="evaluate",
						help="Which mode are we in: [evaluate, gridsearch]")
	parser.add_argument('--dataset', nargs='?', type=str, default='RGBDCows2020',
						help='Which dataset to use: [RGBDCows2020 ,OpenSetCows2020]')
	parser.add_argument('--exclude_difficult', type=int, default=0,
						help='Whether to exclude difficult classes from the loaded dataset')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch size for inference')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='Size of the dense layer for inference')
	parser.add_argument('--current_fold', type=int, default=0,
						help="The current fold we'd like to test on")
	parser.add_argument('--repeat_num', type=int, default=-1,
						help="The repeat number we're on")
	parser.add_argument('--unknown_ratio', type=float, default=-1.0,
						help="The current ratio of unknown classes")
	parser.add_argument('--save_embeddings', type=int, default=1,
						help="Should we save the embeddings to file")
	parser.add_argument('--img_type', type=str, default="RGB",
						help="Which image type should we retrieve: [RGB, D, RGBD]")
	parser.add_argument('--softmax_enabled', type=int, default=1,
						help="Whether softmax was enabled when training the model")

	# Parse them
	args = parser.parse_args()

	if args.mode == "evaluate":
		evaluateModel(args)
	elif args.mode == "gridsearch":
		kGridSearch(args)
	else:
		print(f"Mode not recognised, exiting.")
		sys.exit(1)
