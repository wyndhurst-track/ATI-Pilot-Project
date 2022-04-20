# Core libraries
import os
import sys
import argparse

# Local machine
if os.path.isdir("/home/will"): 
	sys.path.append("/home/will/work/1-RA/src/")
	sys.path.append("/home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network")

# Blue pebble
if os.path.isdir("/home/ca0513"): 
	sys.path.append("/home/ca0513/ATI-Pilot-Project/src/")
	sys.path.append("/home/ca0513/ATI-Pilot-Project/src/Identifier/Supervised-Triplet-Network")
	print("Added paths for blue pebble")

# Blue crystal phase 4
if os.path.isdir("/mnt/storage/home/ca0513"):
	sys.path.append("/mnt/storage/home/ca0513/ATI-Pilot-Project/src/")
	sys.path.append("/mnt/storage/home/ca0513/ATI-Pilot-Project/src/Identifier/Supervisted-Triplet-Network")
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# PyTorch stuff
import torch
from torch import optim
from torch.utils import data
from torch.autograd import Variable

# My libraries
from utils.loss import *
from utils.utils import *
from utils.mining_utils import *


from loader import get_loader, get_data_path
from models.triplet_resnet import *
from models.triplet_resnet_softmax import *
from tqdm import tqdm

from Datasets.OpenSetCows2019 import OpenSetCows2019

"""
File descriptor
"""

# 
def crossValidate(args):
	# Loop through each fold for cross validation
	for k in range(args.num_repeats):
		print(f"Beginning training for repeat {k+1} of {args.num_folds}")

		# Directory for storing data to do with this fold
		store_dir = os.path.join(args.out_path, f"rep_{k}")

		# Change where to store things
		args.ckpt_path = store_dir

		# Create a folder in the results folder for this fold as well as to store embeddings
		os.makedirs(store_dir, exist_ok=True)

		# Let's train!
		trainFold(args, rep=k)

# Train for a single fold
def trainFold(args, rep=0):
	# Load the dataset object
	dataset = OpenSetCows2019(args.unknown_ratio, rep, transform=True, suppress_info=False)
	trainloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=True)

	# Setup/load the relevant model
	if args.softmax_enabled:
		model = triplet_resnet50_softmax(pretrained=True, num_classes=dataset.getNumClasses())
		print("Loaded resnet50 model WITH SOFTMAX")
	else:
		model = triplet_resnet50(pretrained=True, num_classes=dataset.getNumClasses())
		print("Loaded resnet50 model")
	
	# Put it on the GPU
	model.cuda()

	# Initialise the optimiser (DISABLE MOMENTUM IF USING RTL)
	if not args.reciprocal_loss_enabled:
		optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
	else:
		optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)

	# Initialise the loss function
	if args.softmax_enabled:
		if args.reciprocal_loss_enabled:
			loss_fn = OnlineReciprocalSoftmaxLoss(	HardestNegativeTripletSelector(), 
													margin=args.triplet_margin, 
													lambda_factor=0.01	)
			print(f"Using Reciprocal Triplet Loss with softmax")
		else:
			loss_fn = OnlineTripletSoftmaxLoss(	HardestNegativeTripletSelector(), 
												margin=args.triplet_margin, 
												lambda_factor=0.01	)
			print(f"Using Triplet Loss with softmax")
	else:
		# loss_fn = TripletLoss(margin=args.triplet_margin)

		if args.reciprocal_loss_enabled:
			loss_fn = OnlineReciprocalTripletLoss(HardestNegativeTripletSelector())
			print(f"Using Reciprocal Triplet Loss")
		else:
			loss_fn = OnlineTripletLoss(args.triplet_margin, HardestNegativeTripletSelector(args.triplet_margin))
			print(f"Using Triplet Loss")
	
	# Print details about the setup we've employed
	show_setup(args, dataset.getNumClasses(), optimizer, loss_fn)

	# Setup variables
	global_step = 0 
	accuracy_best = 0

	model.train()

	# Main training loop
	for epoch in tqdm(range(args.n_epoch)):
		# Mini-batch training loop over the training set
		for i, (images, images_pos, images_neg, labels, labels_neg) in enumerate(trainloader):
			# Put the images on the GPU and express them as PyTorch variables
			images = Variable(images.cuda())
			images_pos = Variable(images_pos.cuda())
			images_neg = Variable(images_neg.cuda())
		   
			# Zero our optimiser
			optimizer.zero_grad()

			# Get the embeddings/predictions for each
			if args.softmax_enabled:
				embed_anch, embed_pos, embed_neg, preds = model(images, images_pos, images_neg)
			else:
				embed_anch, embed_pos, embed_neg = model(images, images_pos, images_neg)

			# Calculate loss
			if args.softmax_enabled:
				loss, triplet_loss, loss_softmax = loss_fn(	embed_anch, 
															embed_pos, 
															embed_neg, 
															preds, 
															labels,
															labels_neg	)
			else:
				loss = loss_fn(embed_anch, embed_pos, embed_neg, labels)

			# Backprop and optimise
			loss.backward()
			optimizer.step()
			global_step += 1

			# Log the loss
			if global_step % args.logs_freq == 0:
				if args.softmax_enabled:
					log_loss(	epoch, args.n_epoch, global_step, 
								loss_mean=loss.item(), 
								loss_triplet=triplet_loss.item(), 
								loss_softmax=loss_softmax.item()	)
				else:
					log_loss(epoch, args.n_epoch, global_step, loss_mean=loss.item()) 
			
		# Save model weights
		save_checkpoint(epoch, model, optimizer, "temp")

		# Every x epochs, let's evaluate on the validation set
		if epoch % args.eval_freq  == 0:
			# Test on the validation set
			accuracy_curr = eval_model(args.unknown_ratio, rep, global_step, args.instances_to_eval, args.softmax_enabled)

			# Save the model weights as the best if it surpasses the previous best results
			if accuracy_curr > accuracy_best:
				save_checkpoint(epoch, model, optimizer, "best")
				accuracy_best = accuracy_curr

# Main/entry method
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Hyperparams')
	parser.add_argument('--arch', nargs='?', type=str, default='triplet_cnn',
						help='Architecture to use [\'fcn8s, unet, segnet etc\']')
	parser.add_argument('--dataset', nargs='?', type=str, default='open_cows',
						help='Dataset to use [\'cow_id, tless, core50, ade20k etc\']')
	parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--id', nargs='?', type=str, default='x1',
						help='Experiment identifier')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='dense layer size for inference')
	parser.add_argument('--instances', nargs='?', type=str, default='known',
						help='Train Dataset split to use [\'full, known, novel\']')
	parser.add_argument('--instances_to_eval', nargs='?', type=str, default='full',
						help='Test Dataset split to use [\'full, known, novel, all\']')
	parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
						help='# of the epochs')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
						help='Learning Rate')
	parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
						help='Divider for # of features to use')
	parser.add_argument('--ckpt_path', nargs='?', type=str, default='.',
					help='Path to save checkpoints')
	parser.add_argument('--eval_freq', nargs='?', type=int, default=2,
					help='Frequency for evaluating model [epochs num]')
	parser.add_argument('--logs_freq', nargs='?', type=int, default=20,
					help='Frequency for saving logs [steps num]')
	parser.add_argument('--fold_number', type=int, default=0,
						help="The fold number to start at")

	parser.add_argument('--unknown_ratio', type=float, required=True)
	parser.add_argument('--num_repeats', type=int, default=10)


	parser.add_argument('--num_folds', type=int, default=1,
						help="Number of folds to cross validate across")

	parser.add_argument('--out_path', type=str, default="", required=True,
						help="Path to folder to store results in")
	parser.add_argument('--softmax_enabled', type=bool, default=False,
						help="Is softmax enabled for joint loss function")
	parser.add_argument('--reciprocal_loss_enabled', type=bool, default=False,
						help="Is reciprocal loss enabled")
	parser.add_argument('--triplet_margin', type=float, default=0.5,
						help="Margin parameter for triplet loss")

	args = parser.parse_args()

	# Let's train!
	crossValidate(args)
