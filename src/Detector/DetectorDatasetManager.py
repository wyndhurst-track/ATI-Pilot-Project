#!/usr/bin/env python

# Core libraries
import os
import sys
import cv2
import json
import random
#import hickle as hkl
from tqdm import tqdm
import xml.etree.ElementTree as ET

# My libraries
sys.path.append("../")
from config import cfg
from Utilities.Utility import Utility
from Utilities.DataUtils import DataUtils
from Detector.DarknetWrapper import DarknetWrapper
from Visualiser.VisualisationManager import VisualisationManager as VM

"""
This class manages datasets (stored as h5py) for training the detector via darknet
"""

class DetectorDatasetManager():
	# Class constructor
	def __init__(	
					self,
					load_dataset=True
				):
		"""
		Class attributes
		"""

		# The unique counter and IDs for instances in the database
		self.__ID_ctr = 1

		# TEMPORARY: where the dataset h5py file is kept
		self.__dataset_location = "/home/will/work/1-RA/src/Detector/data/detector_dataset.hkl"

		# TEMPORARY: where to write the dataset out to (e.g. in darknet format)
		self.__write_location = "/home/will/work/1-RA/src/Detector/data/consolidated"

		"""
		Class objects
		"""

		"""
		Class setup
		"""

		# Load the dataset
		if load_dataset: self.__loadDataset()

	"""
	Public methods
	"""

	# Add the data from a supplied location to the dictionary
	def importDataFromFolder(self, location):
		# Make sure the supplied location actually exists
		assert os.path.isdir(location)

		# Add the images and labels folders to the path
		images_dir = os.path.join(location, "images")
		labels_dir = os.path.join(location, "labels")

		# Gather the files inside
		image_files = DataUtils.allFilesAtDirWithExt(images_dir, ".jpg")
		label_files = DataUtils.allFilesAtDirWithExt(labels_dir, ".xml")

		# Make sure there are the same amount of files in each
		assert len(image_files) == len(label_files)

		# Loop over the label files and add the instance
		pbar = tqdm(total=len(label_files))
		for i in range(len(label_files)): 
			self.__addInstance(label_files[i], images_dir, image_files, location)
			pbar.update()
		pbar.close()

	# Write the dataset out to file (e.g. images and xml labels) in the required format
	def writeDataset(self, split_data, split_ratio, augment, synthesise_per_img):
		# Ensure the write location exists
		assert os.path.isdir(self.__write_location)

		# Write out the instances from the dict
		image_filenames = self.__writeInstances()

		# If we're supposed to, create a text file with images for training and testing
		# let's split up the data into two parts now so we don't synthesise based on
		# testing samples
		if split_data:
			all_keys = self.__data_dict.keys()

			# Randomly jumble the list
			random.shuffle(all_keys)

			# Determine the index for the split ratio
			split_idx = int(round(len(all_keys)*split_ratio))

			# Split the list of keys into training and testing
			train_keys = all_keys[:split_idx]
			test_keys = all_keys[split_idx:]

		# If we're supposed to, augment training samples
		augmented_filenames = []
		if augment: augmented_filenames = self.__synthesiseInstances(synthesise_per_img, train_keys)

		# If we're supposed to, create a text file with images for training and testing
		if split_data: self.__createSplitFiles(	split_ratio, 
												image_filenames, 
												augmented_filenames,
												train_keys,
												test_keys				)

	# Save the dataset to file (in h5 format via hickle)
	def saveDataset(self):
		print("Writing dataset to file, please wait..")

		# Save
		hkl.dump(self.__data_dict, self.__dataset_location)

		print("Finished writing.")

	"""
	(Effectively) private methods
	"""

	# Load the dataset from file
	def __loadDataset(self):
		# If the file isn't there, it hasn't been created yet
		if not os.path.exists(self.__dataset_location):
			# Initialise the dictionary
			self.__data_dict = {}
		else:
			# Make sure it ends with the correct extension
			assert self.__dataset_location.endswith(".hkl")

			print("Loading dataset, please wait..")

			# Assign the data
			self.__data_dict = hkl.load(self.__dataset_location)

			print("Loaded dataset.")

			# Update the ID counter to match the dataset
			self.__ID_ctr = max(self.__data_dict.keys()) + 1

	# Add a single training instance to the database
	def __addInstance(self, label_file, images_dir, image_files, base_dir):
		# TODO: change this to use the DataUtils method and do this more widely

		# Load the xml file
		tree = ET.parse(label_file)

		# Which jpg filename is this label pointing to, is it in the list?
		image_file = tree.find('filename').text
		if os.path.join(images_dir, image_file) not in image_files: 
			print(os.path.join(images_dir, image_file))
			assert False

		# Load the image into memory
		image = cv2.imread(os.path.join(images_dir, image_file))

		# Dict for this instance
		inst = {}

		# List of objects
		objects = []

		# Loop over all the objects in the label file
		for obj in tree.findall('object'):
			# Get the bounding box data
			x1 = int(obj.find('bndbox').find('xmin').text)
			y1 = int(obj.find('bndbox').find('ymin').text)
			x2 = int(obj.find('bndbox').find('xmax').text)
			y2 = int(obj.find('bndbox').find('ymax').text)

			# Swap components if necessary so that (x1,y1) is always top left
			# and (x2, y2) is always bottom right
			if x1 > x2: x1, x2 = x2, x1
			if y1 > y2: y1, y2 = y2, y1

			# Add this object to the list of objects
			objects.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

		# Make sure there were some annotations, otherwise just skip it
		if len(objects) < 1: print("No annotations for: {}".format(os.path.join(images_dir, image_file)))
		else:
			# Add the objects and image to the instance dictionary
			inst['objects'] = objects
			inst['image'] = image

			# Add this instance to overall database with the correct unique ID, then increment it
			self.__data_dict[self.__ID_ctr] = inst
			self.__ID_ctr += 1

	# Write out instances to file; a .jpg image and a corresponding label .txt file with
	# objects written in the format darknet expects
	def __writeInstances(self):
		print("Writing out instances to file")

		# Maintain a dict of the image filenames for writing out in the train/test files
		# and their ID as the key
		image_filenames = {}

		# Loop through the data dictionary
		pbar = tqdm(total=len(self.__data_dict.keys())-1)
		for k, v in self.__data_dict.items():
			# The key is the image/label filename/ID, convert it to a 6 figure string
			filename = str(k).zfill(6)

			# Append the filename to the directory
			filepath = os.path.join(self.__write_location, filename)

			# Fetch the image and save it, also add it to the list of filenames
			image = v['image']
			cv2.imwrite("{}.jpg".format(filepath), image)
			image_filenames[k] = "{}.jpg".format(filepath)

			# Create the darknet label file for this instance
			text_file = open("{}.txt".format(filepath), "w")

			# Loop over each object in this training instance
			for obj in v['objects']:
				self.__writeDarknetLabelLine(text_file, image.shape, obj['x1'], obj['x2'], obj['y1'], obj['y2'])
			text_file.close()

			pbar.update()
		pbar.close()

		return image_filenames

	# Make an augmentation pass over the data dictionary 'synthesise_per_img' times
	def __synthesiseInstances(self, synthesise_per_img, train_keys):
		print("Commencing augmentation")

		# Make sure the multiplier is an integer
		assert isinstance(synthesise_per_img, int) and synthesise_per_img >= 1

		# Maintain a list of augmented filenames
		augmented_filenames = []

		# Increment a seperate ID counter
		file_ctr = self.__ID_ctr

		# Loop through the data dictionary n times
		pbar = tqdm(total=synthesise_per_img * (len(train_keys)-1))
		for i in range(synthesise_per_img):
			for k in train_keys:
				# Extract the value
				v = self.__data_dict[k]

				# Create a new filename
				filename = str(file_ctr).zfill(6)
				file_ctr += 1

				# Append the filename to the directory
				filepath = os.path.join(self.__write_location, filename)

				# Fetch the image, augment it and write it out
				image, boxes = Utility.augmentImage(v['image'], v['objects'], display=False)
				cv2.imwrite("{}.jpg".format(filepath), image)
				augmented_filenames.append("{}.jpg".format(filepath))

				# Create the darknet label file for this instance
				text_file = open("{}.txt".format(filepath), "w")

				# Loop over each augmented box
				for box in boxes.bounding_boxes: 
					self.__writeDarknetLabelLine(text_file, image.shape, box.x1, box.x2, box.y1, box.y2)
				text_file.close()

				pbar.update()
		pbar.close()

		return augmented_filenames

	# Randomly split image labels into two different .txt files indicating files for training
	# or testing according to a given ratio
	def __createSplitFiles(self, split_ratio, image_filenames, augmented_filenames, train_keys, test_keys):
		# Split the list into training and testing
		train = [image_filenames[x] for x in train_keys]
		test = [image_filenames[x] for x in test_keys]

		# If there are any augmented training files to add, add them to the list and jumble
		# it all up again
		train.extend(augmented_filenames)
		random.shuffle(train)

		# Create and write out the corresponding text files
		train_file = open(os.path.join(self.__write_location, "0train.txt"), "w")
		test_file = open(os.path.join(self.__write_location, "0test.txt"), "w")
		for img in train: train_file.write("{}\n".format(img))
		for img in test: test_file.write("{}\n".format(img))

		train_file.close()
		test_file.close()

	# Compute the darknet label line and write it out to an open text file
	def __writeDarknetLabelLine(self, file, img_shape, x1, x2, y1, y2):
		# Write in darknet format: darknet wants a .txt file for each image with a line for
		# each ground truth object in the image that looks like: 
		# <object-class> <x> <y> <width> <height>

		# Retrieve bounding box values proportional to the image resolution in [0,1]
		dw = 1./img_shape[1]
		dh = 1./img_shape[0]
		x = (x1 + x2)/2.0 - 1
		y = (y1 + y2)/2.0 - 1
		w = x2 - x1
		h = y2 - y1
		x = x * dw
		w = w * dw
		y = y * dh
		h = h * dh

		print(x,y,w,h)

		# Write a line with this cow object
		file.write("0 {} {} {} {}\n".format(x, y, w, h))

	"""
	Getters
	"""

	"""
	Setters
	"""

	"""
	Static methods
	"""

	# Manually view the samples at a given directory
	@staticmethod
	def viewSamples(folder, reverse=False):
		# Get a list of all the images and annotations within the given folder
		images = [os.path.join(folder, x) for x in sorted(os.listdir(folder)) if x.endswith(".jpg")]
		labels = [os.path.join(folder, x) for x in sorted(os.listdir(folder)) if x.endswith(".txt")]

		# If they exist, remove the darknet-required training and testing .txt files
		if os.path.exists(os.path.join(folder, "0test.txt")): labels.remove(os.path.join(folder, "0test.txt"))
		if os.path.exists(os.path.join(folder, "0train.txt")): labels.remove(os.path.join(folder, "0train.txt"))

		# Make sure the lists are equally-sized now
		assert len(images) == len(labels)

		# Should we reverse the orderings of the lists (so that we view augmented samples first)
		if reverse: 
			images.reverse()
			labels.reverse()

		# View them
		pbar = tqdm(total=len(images))
		for i in range(len(images)):
			# Load the image
			cv_image = cv2.imread(images[i])

			# Extract the annotations
			GT = DarknetWrapper.extractDarknetAnnotations(labels[i])

			# Plot them
			VM.plotDetections(cv_image, None, GT_boxes=GT)

			pbar.update()
		pbar.close()

	# Given a folder full of darknet annotations, write out equivalent XML/VOC ones
	@staticmethod
	def darknetToVOC(input_folder, output_folder):
		# Get all the text files in the input folder
		txt_files = DataUtils.allFilesAtDirWithExt(input_folder, "txt")

		# Iterate over them all
		for anno_fp in tqdm(txt_files):
			# Extract the darknet annotation
			data_dict = DataUtils.readDarknetAnnotation(anno_fp)

			# The text file may have contained no actual annotation
			if data_dict is None:
				print(f"Skipping file {os.path.basename(anno_fp)}, contained no annotaion")

			# Otherwise, let's write this out as a XML annotaion
			else:
				# Construct the new filepath
				new_fp = os.path.join(output_folder, os.path.basename(anno_fp)[:-4]+".xml")

				# Write it out in XML form
				DataUtils.writeXMLAnnotation(new_fp, data_dict)

	# The opposite of the method above, convert a folder of VOC to darknet form
	@staticmethod
	def VOCToDarknet(input_folder, output_folder):
		# Get all the XML files in the input folder
		xml_files = DataUtils.allFilesAtDirWithExt(input_folder, "xml")

		# Iterate over them all
		for anno_fp in tqdm(xml_files):
			# Extract the XML annotation
			data_dict = DataUtils.readXMLAnnotation(anno_fp)

			# Construct the new filepath
			new_fp = os.path.join(output_folder, os.path.basename(anno_fp)[:-4]+".txt")

			# Write it out in XML form
			DataUtils.writeDarknetAnnotation(new_fp, data_dict)

	@staticmethod
	def augmentTrainingSamples(fold_file, image_fp, anno_fp):
		""" Augment on the training set of each fold

		Longer desc: tbc
		"""

		# Load the folds file
		with open(fold_file, 'r') as handle:
			fold_dict = json.load(handle)

		# Find all the images
		image_filepaths = DataUtils.allFilesAtDirWithExt(image_fp, ".jpg")

		# Iterate through each fold
		for k in fold_dict.keys():
			train_len = len(fold_dict[k]['train'])
			valid_len = len(fold_dict[k]['valid'])
			test_len = len(fold_dict[k]['test'])

			assert len(image_filepaths) == train_len + valid_len + test_len

			# Iterate through each training image
			print(len(fold_dict[k]['train']))

# Entry method/unit testing method
if __name__ == '__main__':
	"""
	Enrolling/importing new data into the dataset
	"""

	# # Dir of the folder to be imported
	# import_folders = [	
	# 					"/home/will/work/1-RA/src/Detector/data/raw/Sion-Colin-Combined",
	# 					"/home/will/work/1-RA/src/Detector/data/raw/BBC/1",
	# 					"/home/will/work/1-RA/src/Detector/data/raw/BBC/2",
	# 					"/home/will/work/1-RA/src/Detector/data/raw/BBC/3",
	# 					"/home/will/work/1-RA/src/Detector/data/raw/FlightExperiments"
	# 				 ]

	# # Import a folder of images and labels
	# for folder in import_folders:
	# 	print "Importing from: {}".format(folder)
	# 	manager.importDataFromFolder(folder)

	# # Let's save it!
	# manager.saveDataset()

	"""
	Writing out the dataset to file (e.g. in darknet format)
	"""

	# # Create an instance
	# manager = DetectorDatasetManager(0)

	# # Whether or not to also write out text file with train/test files and if so, the train
	# # test split ratio
	# split_data = False
	# split_ratio = 0.9

	# # Whether to augment training samples for additional data during training (not testing)
	# # and the number of images to synthesise per natural image (must be an integer)
	# augment = False
	# synthesise_per_img = 1

	# # Write it!
	# manager.writeDataset(split_data, split_ratio, augment, synthesise_per_img)

	"""
	View samples and bounding boxes from the darknet format
	"""

	# folder = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\darknet"
	# DetectorDatasetManager.viewSamples(folder, reverse=False)

	"""
	Convert a set of darknet annotations to (Pascal/VOC) XML form
	"""

	# input_folder = "D:\\Work\\Data\\CEADetection\\darknet"
	# output_folder = "D:\\Work\\Data\\CEADetection\\VOC"
	# DetectorDatasetManager.darknetToVOC(input_folder, output_folder)

	"""
	Convert a set of Pascal/VOC XML annotations to darknet form
	"""

	# input_folder = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\VOC"
	# output_folder = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\darknet"
	# DetectorDatasetManager.VOCToDarknet(input_folder, output_folder)

	base_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection"
	fold_file = os.path.join(base_dir, "10-fold-CV.json")
	image_fp = os.path.join(base_dir, "images")
	anno_fp = os.path.join(base_dir, "labels-xml")
	DetectorDatasetManager.augmentTrainingSamples(fold_file, image_fp, anno_fp)