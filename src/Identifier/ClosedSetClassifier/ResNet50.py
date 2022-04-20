# Core libraries
import os
import sys
if os.path.isdir("/home/will"): sys.path.append("/home/will/work/1-RA/src/")
if os.path.isdir("D:\\Work"): sys.path.append("D:\\Work\\ATI-Pilot-Project\\src")
if os.path.isdir("/home/ca0513"): sys.path.append("/home/ca0513/ATI-Pilot-Project/src/")
if os.path.isdir("/mnt/storage/home/ca0513"): sys.path.append("/mnt/storage/home/ca0513/ATI-Pilot-Project/src/")
import numpy as np
import json
import pickle
import argparse
import copy
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# PyTorch stuff
import torch
from torch.utils import data
from torch import optim
from torchvision import transforms, datasets, models

# My libraries
from Datasets.OpenSetCows2019 import OpenSetCows2019
from Datasets.RGBDCows2020 import RGBDCows2020

class LateFusionModel(torch.nn.Module):
	"""
	Dual-stream ResNet50 model performing late fusion RGB & D fusion
	"""
	def __init__(self, num_classes):
		# Initialise super class
		super(LateFusionModel, self).__init__()

		# Individual ResNet models pretrained on imagenet
		self.__RGB_Net = models.resnet50(pretrained=True)
		self.__D_Net = models.resnet50(pretrained=True)

		RGB_out_features = self.__RGB_Net.fc.out_features
		D_out_features = self.__D_Net.fc.out_features

		self.__fc = torch.nn.Linear(RGB_out_features+D_out_features, num_classes)

	def forward(self, RGB, D):
		x1 = self.__RGB_Net(RGB)
		x2 = self.__D_Net(D)

		x = torch.cat((x1,x2), dim=1)
		x = self.__fc(x)

		return x

def crossValidate(args):
	"""
	Performs cross-fold validation
	"""

	# Dictionary to store training information
	data_logs = {}

	# Which dataset split mode are we in?
	if args.split_mode == "random":
		start = args.start_fold
		stop = args.end_fold+1 if args.end_fold >= 0 else args.num_folds
		step = 1
	elif args.split_mode == "day":
		start = 3
		stop = 31 - 3
		step = 3

	# Iterate through folds or days
	for k in range(start, stop, step):
		# Which dataset are we loading
		if args.dataset == "RGBDCows2020":
			# Initialise the dataset based on which split mode we're in
			if args.split_mode == "random":
				# Load our dataset for this fold
				train_dataset = RGBDCows2020(fold=k, split="train", transform=True, img_type=args.img_type, depth_type=args.depth_type)
				valid_dataset = RGBDCows2020(fold=k, split="test", transform=True, img_type=args.img_type, depth_type=args.depth_type)

			elif args.split_mode == "day":
				# Load these selections of days
				train_dataset = RGBDCows2020(split_mode="day", num_training_days=k, split="train", transform=True, img_type=args.img_type, depth_type=args.depth_type)
				valid_dataset = RGBDCows2020(split_mode="day", num_testing_days=step, split="test", transform=True, img_type=args.img_type, depth_type=args.depth_type)
				print(f"Loaded {k} training days, {step} testing days")
		elif args.dataset == "OpenSetCows2019":
			# Load the training and validation sets
			train_dataset = OpenSetCows2019(args.unknown_ratio, k, split="train", transform=True, suppress_info=False)
			valid_dataset = OpenSetCows2019(args.unknown_ratio, k, split="valid", transform=True, suppress_info=False, combine=True)

		# Retrieve the cardinality of the datasets
		dataset_sizes = {'train': len(train_dataset), 'val': len(valid_dataset)}

		# Put our data into a pytorch-friendly loader
		dataloader = {}
		dataloader['train'] = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
		dataloader['val'] = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

		# The number of classes in our dataset
		assert train_dataset.getNumClasses() == valid_dataset.getNumClasses()
		num_classes = train_dataset.getNumClasses()

		print(f"Found {num_classes} distinct classes")

		# Create our model pretrained in ImageNet
		model = models.resnet50(pretrained=True)

		# Change the number of output neurons of the last layer to the number of classes we're expecting
		model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

		# Are we using RGB and Depth imagery?
		if args.img_type == "RGBD":
			# When are we fusing RGB and D, if early, then just change the number of channels the first layer
			# is expecting
			if args.fusion_method == "early":
				model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
			# Create a dual stream network and fuse at the end of the network
			elif args.fusion_method == "late":
				model = LateFusionModel(num_classes)
			else:
				print(f"Didn't recognise fusion method, possibilities = [early, late]")
				sys.exit(1)

		# Train the model
		data_logs_fold = trainModel(	k, 
										model, 
										dataloader, 
										dataset_sizes, 
										args.num_epochs, 
										args.out_path, 
										args.img_type, 
										args.fusion_method,
										args.batch_size			)

		# Path to file
		logs_filepath = os.path.join(args.out_path, f"rep_{k}_data_logs.pkl")

		# Save data logs out to do with this fold
		with open(logs_filepath, "wb") as handle:
			pickle.dump(data_logs_fold, handle)

def trainModel(k, model, dataloader, dataset_sizes, num_epochs, out_path, img_type, fusion_method, batch_size):
	"""
	Train ResNet50 model for a particular fold

	**Parameters**
	> **k:** `int` -- Current fold number.
	> **model:** `` -- PyTorch model we want to train. Potentially with pretrained weights.
	> **dataloader:** `dict` -- Dictionary of training and testing datasets loaded using the PyTorch data loader.
	> **dataset_sizes:** `dict` -- Dictionary of dataset cardinality for training and testing.
	> **num_epochs:** `int` -- Number of epochs to train for.
	> **out_path:** `str` -- Absolute path to directory for storing network weights and logs.
	> **batch_size:** `int` -- Batch size.

	**Returns**
	> `dict` -- Data logs from training organised into loss and accuracy for training and validation.
	"""

	best_model_weights = copy.deepcopy(model.state_dict())
	best_accuracy = 0.0

	# Put the model on the GPU
	model.cuda()

	# Optimiser
	optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	# Learning rate scheduler
	scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)

	# Our loss function
	loss_fn = torch.nn.CrossEntropyLoss()

	# Define our device
	assert torch.cuda.is_available()
	device = torch.device("cuda:0")

	# Storage for data logs
	data_logs = {}
	data_logs['loss'] = {'train': [], 'val': []}
	data_logs['acc'] = {'train': [], 'val': []}
	global_step = 0

	# Let's train!
	for epoch in tqdm(range(num_epochs), desc="Training epochs"):
		print(f'Epoch {epoch}/{num_epochs-1}')
		print('-' * 10)

		# Train for an epoch, then test
		for phase in ['train', 'val']:
			if phase == 'train': model.train()
			else: model.eval()

			running_loss = 0.0
			running_corrects = 0

			# Iterate through the training or testing set
			for images, _, _, labels, _ in dataloader[phase]:
			# for images, labels, _ in dataloader[phase]:
				# We need to separate the batch into batches of RGB and D
				if img_type == "RGBD" and fusion_method == "late":
					RGB = images[:,:3,:,:]
					D = torch.zeros(RGB.shape)
					D[:,0,:,:] = images[:,3,:,:]
					D[:,1,:,:] = images[:,3,:,:]
					D[:,2,:,:] = images[:,3,:,:]
					
					# Put them on the GPU
					RGB = RGB.to(device)
					D = D.to(device)
				else:
					# Put the images and labels on the GPU
					inputs = images.to(device)

				# Flatten the labels into a 1D tensor for the cross entropy loss function and GPU them
				labels = torch.flatten(labels-1)
				labels = labels.to(device)

				# Zero the optimiser
				optimiser.zero_grad()

				with torch.set_grad_enabled(phase=='train'):
					# Get logits for this batch
					if img_type == "RGBD" and fusion_method == "late":
						logits = model(RGB, D)
					else: logits = model(inputs)
					
					# Compute loss over this batch
					loss = loss_fn(input=logits, target=labels)

					# Normalise the logits via softmax
					softmax = torch.nn.functional.softmax(logits, dim=1)

					# Get the predictions from this
					_, preds = torch.max(softmax.data, 1)

					if phase == 'train':
						loss.backward()
						optimiser.step()

				running_loss += loss.item() * batch_size
				running_corrects += torch.sum(preds == labels.data)

				global_step += 1
				if global_step % 100 == 0:
					print(f"Global step: {global_step}, batch loss: {loss.item()}")

					with open(os.path.join(args.out_path, f"rep_{k}_data_logs.pkl"), "wb") as handle:
						pickle.dump(data_logs, handle)

			if phase == 'train': scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			data_logs['loss'][phase].append(epoch_loss)
			data_logs['acc'][phase].append(float(epoch_acc.cpu().numpy()))

			print(f"{phase} loss: {epoch_loss}, acc: {epoch_acc}")

			if phase == 'val' and epoch_acc > best_accuracy:
				best_accuracy = epoch_acc

				# Save these best weights out
				best_model_weights = copy.deepcopy(model.state_dict())
				save_path = os.path.join(args.out_path, f"rep_{k}_best_model_weights.pth")
				torch.save(best_model_weights, save_path)

	print(f"Best accuracy: {best_accuracy}")

	return data_logs

def testModel(args):
	"""
	Function description
	"""

	# Aggregate testing statistics across each fold
	stats = {}

	# Keep a long list of predicted and ground truth labels across all folds
	predicted = []
	ground_truth = []

	# Iterate through each fold
	for fold in range(args.start_fold, args.num_folds):
		if args.dataset == "RGBDCows2020":
			# Get the testing set object
			test_dataset = RGBDCows2020(fold=fold, split="test", transform=True, suppress_info=False, img_type=args.img_type)
		elif args.dataset == "OpenSetCows2019":
			test_dataset = OpenSetCows2019(args.unknown_ratio, fold, split="valid", combine=True, transform=True, suppress_info=False)

		# Load in PyTorch form
		test_set = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

		# Use the GPU if we can
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Get the model definition
		model = models.resnet50()

		# Change the last layer to match the number of classes we're expecting
		model.fc = torch.nn.Linear(model.fc.in_features, test_dataset.getNumClasses())

		# What image type are we using
		img_type = args.img_type

		# Are we using RGB and Depth imagery?
		if img_type == "RGBD":
			# When are we fusing RGB and D, if early, then just change the number of channels the first layer
			# is expecting
			if args.fusion_method == "early":
				model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
			# Create a dual stream network and fuse at the end of the network
			elif args.fusion_method == "late":
				img_type += "_late_fusion"
				model = LateFusionModel(test_dataset.getNumClasses())
			else:
				print(f"Didn't recognise fusion method, possibilities = [early, late]")
				sys.exit(1)

		# Where to load weights from
		# base_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Results\\ClosedSet"
		# model_path = os.path.join(base_dir, f"{img_type}\\fold_{fold}_best_model_weights.pth")

		base_dir = "/work/ca0513/results/CEiA/ClosedSet/09"
		model_path = os.path.join(base_dir, f"rep_{fold}_best_model_weights.pth")
		
		assert os.path.exists(model_path)

		# Load weights from file
		weights = torch.load(model_path)
		model.load_state_dict(weights)

		# Put the model in evaluation mode and on the selected device
		model.eval()
		model.to(device)

		correct = 0
		total = len(test_dataset)

		# Keep stats about predictions
		fold_stats = {i: {'correct': [], 'incorrect': []} for i in range(test_dataset.getNumClasses())}

		# Iterate through the dataset
		# for images, labels, filepaths in tqdm(test_set):
		for images, _, _, labels, _ in tqdm(test_set):
			# We don't need the autograd engine
			with torch.no_grad():
				# Flatten the labels into a 1D tensor for the cross entropy loss function and GPU them
				labels = torch.flatten(labels).to(device)

				# We need to separate the batch into batches of RGB and D
				if img_type == "RGBD_late_fusion":
					RGB = images[:,:3,:,:]
					D = torch.zeros(RGB.shape)
					D[:,0,:,:] = images[:,3,:,:]
					D[:,1,:,:] = images[:,3,:,:]
					D[:,2,:,:] = images[:,3,:,:]
					
					# Put them on the GPU
					RGB = RGB.to(device)
					D = D.to(device)

					# Get output from the network
					logits = model(RGB, D)
				else:
					# Put the images and labels on the GPU
					inputs = images.to(device)

					# Get output from the network
					logits = model(inputs)

				# Normalise the logits via softmax
				softmax = torch.nn.functional.softmax(logits, dim=1)

				# Get the predictions from this
				_, preds = torch.max(softmax.data, 1)

				# Store the predictions and actual labels
				ground_truth.extend(labels.cpu().numpy().tolist())
				predicted.extend(preds.cpu().numpy().tolist())

				# Go through and mark prediction stats
				assert preds.shape == labels.data.shape
				for j in range(preds.shape[0]):
					# Extract the current label
					current_class = int(labels.data[j].cpu().numpy())

					# Extract the current file we tested
					# filename = filepaths[j]

					# Were we correct or wrong
					if preds[j] == labels.data[j] - 1: 
						# fold_stats[current_class]['correct'].append(filename)
						correct += 1
					# else: fold_stats[current_class]['incorrect'].append(filename)

		# Force GPU memory to be freed
		del model

		# Add stats about this fold to the larger dictionary
		stats[fold] = fold_stats

		print(f"Rep {fold} got {correct}/{total} = {float(correct)/total*100}% correct")

	# Add predicted and GT labels
	stats['ground_truth'] = ground_truth
	stats['predicted'] = predicted

	# Save the statistics out
	# out_filepath = os.path.join(args.out_path, f"{img_type}_testing_stats.json")
	# with open(out_filepath, 'w') as handle:
	# 	json.dump(stats, handle, indent=4)

	# print(f"Wrote out testing stats to: {out_filepath}")

def analyse(args):
	"""
	Perform basic analysis on saved testing data
	"""

	# Extract the image type
	img_type = args.img_type

	# Modify it if we're in late fusion mode
	if img_type == "RGBD" and args.fusion_method == "late": img_type += "_late_fusion"

	# Load the relevant data
	filepath = os.path.join(args.out_path, f"{args.img_type}_testing_stats.json")
	with open(filepath, 'r') as handle:
		stats = json.load(handle)

	# Print fold/ID/filenames of incorrectly predicted test instances
	# for i in range(args.num_folds):
	# 	fold_stats = stats[str(i)]
	# 	print(f"Fold = {i}:")
	# 	for k, v in fold_stats.items():
	# 		num_correct = len(v['correct'])
	# 		num_incorrect = len(v['incorrect'])
	# 		if num_incorrect > 0:
	# 			print(f"    ID = {k} had {num_correct} correct and {num_incorrect} incorrect predictions. They were:")
	# 			for filename in v['incorrect']: print(f"        {filename}")

	# Determine which classes had the highest error rate across all folds
	# error_rates = {k: {'correct': 0, 'incorrect': 0} for k in stats['0'].keys()}
	# for i in range(args.num_folds):
	# 	for k, v in stats[str(i)].items():
	# 		error_rates[k]['correct'] += len(v['correct'])
	# 		error_rates[k]['incorrect'] += len(v['incorrect'])

	# for k, v in error_rates.items():
	# 	error_rate = float(v['incorrect']) / (v['incorrect'] + v['correct'])
	# 	error_rates[k]['error_rate'] = error_rate
	# 	if error_rate > 0:
	# 		print(f"ID = {k}, error rate = {error_rate}")

	# Produce confusion matrix - based on:
	# https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
	cm = confusion_matrix(stats['ground_truth'], stats['predicted'], normalize='all')
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.rainbow)
	plt.title("Confusion Matrix")
	plt.colorbar()
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	# plt.show()
	plt.savefig(os.path.join(args.out_path, f"{img_type}_confusion_matrix.pdf"))

def dispBestAccuracies(file_path):
	"""
	Function descriptor
	"""

	with open(file_path, "rb") as handle:
		logs = pickle.load(handle)

	print(f"File: {file_path}")
	
	# Iterate over each fold
	for k, data in logs.items():
		best_acc = np.max(data['acc']['val']) * 100
		print(f"Best accuracy for {k}: {best_acc}")

# Main/entry function
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters")

	# Required arguments
	parser.add_argument('--out_path', type=str, required=True)

	# Non-required arguments
	parser.add_argument('--dataset', type=str, default="RGBDCows2020",
						help="Which dataset to train on: [RGBDCows2020, OpenSetCows2019]")
	parser.add_argument('--mode', type=str, default="train",
						help="Which mode we're in: [train, test, analysis]")
	parser.add_argument('--split_mode', type=str, default="random",
						help="How to split the dataset into train/test: [random, day]")
	parser.add_argument('--start_fold', type=int, default=0)
	parser.add_argument('--end_fold', type=int, default=-1)
	parser.add_argument('--img_type', type=str, default="RGB")
	parser.add_argument('--depth_type', type=str, default="normal")
	parser.add_argument('--num_folds', type=int, default=10)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--fusion_method', type=str, default="early",
						help="If using RGBD, when to perform fusion [early, late]")

	# Stuff for CEiA paper
	parser.add_argument('--unknown_ratio', type=float, required=True)

	args = parser.parse_args()

	if args.mode == "train":
		crossValidate(args)
	elif args.mode == "test":
		testModel(args)
	elif args.mode == "analysis":
		analyse(args)
