#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import cv2
import math
import json
import pickle
import random
import numpy as np
from PIL import Image

# pyTorch
import torch
from torch.utils import data

# My libraries
from config import cfg
from Utilities.DataUtils import DataUtils
from Utilities.ImageUtils import ImageUtils

class OpenSetCows2019(data.Dataset):
	"""
	Class manages i/o for the OpenSetCows2020 dataset in PyTorch
	"""

	def __init__(	self,
					unknown_ratio,
					n,
					split="train",
					combine=False,
					known=True,
					transform=False,
					img_size=(224, 224),
					suppress_info=True	):
		"""
		Class constructor
		"""

		# The root directory for the dataset itself
		self.__root = cfg.DATASET.OPENSETCOWS2020_LOC

		# There are two files we need to load that tell us how to split the instances into training/testing/validation
		# and how to split the categories into known/unknown
		with open(os.path.join(self.__root, "known_unknown_splits.json")) as handle:
			self.__category_splits = json.load(handle)
		with open(os.path.join(self.__root, "train_valid_test_splits.json")) as handle:
			self.__instance_splits = json.load(handle)

		# The desired ratio of unknown to known categories
		self.__unknown_ratio = str(unknown_ratio)

		# The categories are randomly split 10 times, which repeat are we
		self.__n = n

		# The instance split we're after (e.g. train/valid/test)
		self.__split = split

		# Whether we should combine known and unknown categories
		self.__combine = combine

		# Whether we're after known or unknown categories
		self.__known = known

		# Whether to transform images/labels into pyTorch form
		self.__transform = transform

		# Get the list of folders/categories
		self.__category_filepaths = DataUtils.allFoldersAtDir(self.__root)

		# Get the number of classes from this
		self.__num_classes = len(self.__category_filepaths)

		# The image size to resize to
		self.__img_size = img_size

		# A dictionary storing seperately the list of image filepaths per category for 
		# training, validation and testing
		self.__sorted_files = {'train': {}, 'valid': {}, 'test': {}}

		# A dictionary storing separately the complete lists of filepaths for training, validation and testing
		self.__files = {'train': [], 'valid': [], 'test': []}

		"""
		Class setup
		"""

		# We should return both known and unknown categories
		if self.__combine:
			categories = self.__category_splits[self.__unknown_ratio][n]['known'] + \
						 self.__category_splits[self.__unknown_ratio][n]['unknown']
		# We want either just the known or just the unkown categories
		else:
			if self.__known: categories = self.__category_splits[self.__unknown_ratio][n]['known']
			else: categories = self.__category_splits[self.__unknown_ratio][n]['unknown']

		# Load from these acceptable/visible categories
		for category in categories:
			# Populate the large lists of files
			self.__files['train'].extend(self.__instance_splits[category]['train'])
			self.__files['valid'].extend(self.__instance_splits[category]['valid'])
			self.__files['test'].extend(self.__instance_splits[category]['test'])

			# Populate the dictionary with sorted by category
			self.__sorted_files['train'][category] = self.__instance_splits[category]['train']
			self.__sorted_files['valid'][category] = self.__instance_splits[category]['valid']
			self.__sorted_files['test'][category] = self.__instance_splits[category]['test']

		# Report some things
		if not suppress_info: self.printStats()

	"""
	Superclass overriding methods
	"""

	def __len__(self):
		"""
		Get the number of items for this dataset (depending on the split)
		"""
		return len(self.__files[self.__split])

	def __getitem__(self, index):
		"""

		"""

		# Get and load the anchor image
		anchor_filename = self.__files[self.__split][index]
		
		# Retrieve the class/label this index refers to
		current_category = self.__retrieveCategoryForFilepath(anchor_filename)

		# Formulate the complete path
		anchor_path = os.path.join(self.__root, current_category, anchor_filename)

		# Load the anchor image
		img_anchor = self.__loadResizeImage(anchor_path)
		
		# Get a positive (another random image from this class)
		img_pos = self.__retrievePositive(current_category, anchor_filename)

		# Get a negative (a random image from a different random class)
		img_neg, label_neg = self.__retrieveNegative(current_category, anchor_filename)

		# Convert all labels into numpy form
		label_anchor = np.array([int(current_category)])
		label_neg = np.array([int(label_neg)])

		# For sanity checking, visualise the triplet
		# if self.__split == "test":
		# self.__visualiseTriplet(img_anchor, img_pos, img_neg, label_anchor)

		# Transform to pyTorch form
		if self.__transform:
			img_anchor, img_pos, img_neg = self.__transformImages(img_anchor, img_pos, img_neg)
			label_anchor, label_neg = self.__transformLabels(label_anchor, label_neg)

		return img_anchor, img_pos, img_neg, label_anchor, label_neg

	"""
	Public methods
	"""	

	# Print stats about the current state of this dataset
	def printStats(self):
		# Extract the number of known and unknown classes
		num_known = len(self.__category_splits[self.__unknown_ratio][self.__n]['known'])
		num_unknown = len(self.__category_splits[self.__unknown_ratio][self.__n]['unknown'])

		# Extract the number of train, valid, test images
		num_train = len(self.__files['train'])
		num_valid = len(self.__files['valid'])
		num_test = len(self.__files['test'])

		print("Loaded the OpenSetCows2019 dataset_____________________________")
		print(f"Unknown ratio={self.__unknown_ratio}, split={self.__split}, combine={self.__combine}, known={self.__known}, repeat={self.__n}")
		print(f"Found {self.__num_classes} categories: {num_known} known, {num_unknown} unknown")
		print(f"With {num_train} train images, {num_valid} valid images, {num_test} test images")
		print(f"Unknown categories: {self.__category_splits[self.__unknown_ratio][self.__n]['unknown']}")
		print("_______________________________________________________________")

	"""
	(Effectively) private methods
	"""

	def __visualiseTriplet(self, image_anchor, image_pos, image_neg, label_anchor):
		print(f"Label={label_anchor}")
		cv2.imshow(f"Label={label_anchor} anchor", image_anchor)
		cv2.imshow(f"Label={label_anchor} positive", image_pos)
		cv2.imshow(f"Label={label_anchor} negative", image_neg)
		cv2.waitKey(0)

	# Transform the numpy images into pyTorch form
	def __transformImages(self, img_anchor, img_pos, img_neg):
		# Firstly, transform from NHWC -> NCWH
		img_anchor = img_anchor.transpose(2, 0, 1)
		if img_pos is not None: img_pos = img_pos.transpose(2, 0, 1)
		img_neg = img_neg.transpose(2, 0, 1)

		# Now convert into pyTorch form
		img_anchor = torch.from_numpy(img_anchor).float()
		if img_pos is not None: img_pos = torch.from_numpy(img_pos).float()
		img_neg = torch.from_numpy(img_neg).float()

		return img_anchor, img_pos, img_neg

	# Transform the numpy labels into pyTorch form
	def __transformLabels(self, label_anchor, label_neg):
		# Convert into pyTorch form
		label_anchor = torch.from_numpy(label_anchor).long()
		label_neg = torch.from_numpy(label_neg).long()

		return label_anchor, label_neg

	# Print some info about the distribution of images per category
	def __printImageDistribution(self):
		for category, filepaths in self.__sorted_files[self.__split].items():
			print(category, len(filepaths))

	# For a given filepath, return the category which contains this filepath
	def __retrieveCategoryForFilepath(self, filepath):
		# Iterate over each category
		for category, filepaths in self.__sorted_files[self.__split].items():
			if filepath in filepaths: return category

	# Get another positive sample from this class
	def __retrievePositive(self, category, filepath):
		# Copy the list of possible positives and remove the anchor
		possible_list = list(self.__sorted_files[self.__split][category])
		assert filepath in possible_list

		# There may just be one image for this category/split, if so just return it
		if len(possible_list) > 1: possible_list.remove(filepath)

		# Randomly select a filepath
		filename = random.choice(possible_list)

		# Complete the path to it
		img_path = os.path.join(self.__root, category, filename)

		# Load and return the image
		img = self.__loadResizeImage(img_path)
		return img

	def __retrieveNegative(self, category, filepath):
		# Get the list of categories and remove that of the anchor
		possible_categories = list(self.__sorted_files[self.__split].keys())
		assert category in possible_categories
		possible_categories.remove(category)

		# Randomly select a category
		random_category = random.choice(possible_categories)

		# Randomly select a filepath in that category
		filename = random.choice(self.__sorted_files[self.__split][random_category])

		# Complete the path to it
		img_path = os.path.join(self.__root, random_category, filename)

		# Load and return the image along with the selected label
		img = self.__loadResizeImage(img_path)
		return img, random_category

	# Load an image into memory and resize it as required
	def __loadResizeImage(self, img_path):
		# # Load the image
		# img = cv2.imread(img_path)

		# # Resize it proportionally to a maximum of img_size
		# img = ImageUtils.proportionallyResizeImageToMax(img, self.__img_size[0], self.__img_size[1])

		# pos_h = int((self.__img_size[0] - img.shape[0])/2)
		# pos_w = int((self.__img_size[1] - img.shape[1])/2)

		# # Paste it into an zeroed array of img_size
		# new_img = np.zeros((self.__img_size[0], self.__img_size[1], 3), dtype=np.uint8)
		# new_img[pos_h:pos_h+img.shape[0], pos_w:pos_w+img.shape[1], :] = img

		img = Image.open(img_path)

		old_size = img.size

		ratio = float(self.__img_size[0])/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])

		img = img.resize(new_size, Image.ANTIALIAS)

		new_img = Image.new("RGB", (self.__img_size[0], self.__img_size[1]))
		new_img.paste(img, ((self.__img_size[0]-new_size[0])//2,
					(self.__img_size[1]-new_size[1])//2))

		new_img = np.array(new_img, dtype=np.uint8)

		return new_img

	"""
	Getters
	"""

	def getNumClasses(self):
		return self.__num_classes

	def getNumTrainingFiles(self):
		return len(self.__files["train"])

	def getNumTestingFiles(self):
		return len(self.__files["test"])

	"""
	Setters
	"""

	"""
	Static methods
	"""

	@staticmethod
	def splitIntoKnownUnknown(dataset_location, output_dir, num_random_splits=10):
		"""
		Split the categories randomly into known and unknown, sampling equally from outdoor and indoor categories
		"""

		# Get a list of the categories
		categories = DataUtils.allFoldersAtDir(dataset_location, full_path=False)

		# Lists of categories sorted by source (in order: IROS 2019 paper, ICIP 2016 paper, ICCVW 2017 paper)
		outdoor_cats1 = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010",\
						 "011", "012", "013", "014", "015", "016", "017"]
		indoor_cats = 	["018", "019", "020", "021", "022", "023", "024", "025", "026", "027",\
						 "028", "029", "030", "031", "032", "033", "034", "035", "036", "037",
						 "038", "039"]
		outdoor_cats2 = ["040", "041", "042", "043", "044", "045", "046"]
		outdoor_cats = outdoor_cats1 + outdoor_cats2
		assert len(categories) == len(outdoor_cats1) + len(indoor_cats) + len(outdoor_cats2)

		# The list of unknown ratios of categories we want to experiment with on open-set
		ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.33, 0.25, 0.17, 0.1]

		print(f"Found {len(indoor_cats)} indoor categories, {len(outdoor_cats1)+len(outdoor_cats2)} outdoors.")

		split_dict = {}

		# Iterate through each unknown ratio
		for unknown_ratio in ratios:
			# Compute the number of unknown and known categories we need for this ratio
			num_unknown = math.floor(unknown_ratio * len(categories))
			num_known = len(categories) - num_unknown

			# Determine the whole numbers of categories to sample equally from indoor and outdoor
			num_unknown_indoor = math.floor(num_unknown/2)
			num_unknown_outdoor = num_unknown - num_unknown_indoor

			# Create the dictionary element for this ratio
			split_dict[unknown_ratio] = []

			# Randomly split the categories in this ratio (maintaining equal indoor/outdoor sampling)
			for i in range(num_random_splits):
				# Make a copy of the list of indoor and outdoor categories
				indoor_copy = list(indoor_cats)
				outdoor_copy = list(outdoor_cats)

				# Randomly shuffle the categories
				random.shuffle(indoor_copy)
				random.shuffle(outdoor_copy)

				# Choose the first n from each list to be our list of unknown categories
				unknown_cats = indoor_copy[:num_unknown_indoor] + outdoor_copy[:num_unknown_outdoor]

				# Make the rest the known categories
				known_cats = [x for x in categories if x not in unknown_cats]

				# Add these to the dict
				split_dict[unknown_ratio].append({'known': known_cats, 'unknown': unknown_cats})

		# Save out to a json file
		with open(os.path.join(output_dir, "known_unknown_splits.json"), 'w') as handle:
			json.dump(split_dict, handle, indent=1)

	@staticmethod
	def extractErroneousRegions(dataset_location, visualise=False):
		# Define the errors statically (for now) (image ID: bbox)

		# Definite errors
		# errors = {	590: [1049,742,437,290],
		# 			743: [924,96,558,310],
		# 			2427: [23,422,132,132],
		# 			2538: [415,425,102,62],
		# 			2581: [432,168,98,51],
		# 			2584: [53,169,107,71],
		# 			2607: [0,219,84,41],
		# 			2623: [641,419,88,48],
		# 			2719: [471,0,184,57],
		# 			2730: [604,262,128,74],
		# 			2967: [388,262,119,202],
		# 			3169: [347,350,56,153],
		# 			3200: [284,199,82,146],
		# 			3203: [556,37,65,144],
		# 			3312: [440,4,293,235]		}

		# Possible errors
		errors = {	162: [893,38,575,279],
					1140: [516,167,581,455],
					1538: [245,0,461,333],
					1803: [69,547,534,166],
					2654: [175,347,130,48],
					2654: [161,399,118,57],
					3122: [56,165,95,96],
					3200: [284,199,82,146],
					3200: [352,206,60,135],
					3530: [95,363,172,360]		}

		# Iterate through each key
		for image_id, box in errors.items():
			# Load the image this entry refers to
			image_path = os.path.join(dataset_location, str(image_id).zfill(6)+".jpg")
			image = cv2.imread(image_path)

			# Extract the RoI
			# x1 = box[0]
			# y1 = box[1]
			# x2 = box[2]
			# y2 = box[3]
			x1 = int(box[0] - box[2]/2)
			y1 = int(box[1] - box[3]/2)
			x2 = x1 + int(box[2])
			y2 = y1 + int(box[3])

			# Clamp any negative values at zero
			if x1 < 0: x1 = 0
			if y1 < 0: y1 = 0
			if x2 < 0: x2 = 0
			if y2 < 0: y2 = 0

			# Swap components if necessary so that (x1,y1) is always top left
			# and (x2, y2) is always bottom right
			if x1 > x2: x1, x2 = x2, x1
			if y1 > y2: y1, y2 = y2, y1

			RoI = image[y1:y2,x1:x2]

			# Display it if we're supposed to
			if visualise:
				# cv2.rectangle(image, (x1,y1),(x2,y2), (255,0,0), 3)
				# cv2.imshow("Extracted region", RoI)
				# cv2.waitKey(0)
				pass

			# Construct path and save
			save_path = os.path.join(output_dir, str(image_id).zfill(6)+".jpg")
			cv2.imwrite(save_path, RoI)

# Entry method/unit testing method
if __name__ == '__main__':
	# Create a dataset instance
	# dataset = OpenSetCows2019(0.9, 0)
	# for i in range(len(dataset)):
	# 	test = dataset[i]

	# From a list of erroneous detections (e.g. from RetinaNet)
	# dataset_location = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\darknet"
	# output_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\RetinaNet-failures\\possible"
	# OpenSetCows2019.extractErroneousRegions(dataset_location, output_dir)
	
	# Split the categories into known/unknown
	dataset_location = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\OpenCows2020\\identification\\images"
	output_dir = ""
	OpenSetCows2019.splitIntoKnownUnknown(dataset_location, output_dir)
