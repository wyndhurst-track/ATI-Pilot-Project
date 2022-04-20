# Core libraries
import os
import sys
sys.path.append("../../../")
import cv2

# My libraries
from Utilities.DataUtils import DataUtils

"""
Class ...
"""
class FileHandler():
	# Class constructor
	def __init__(self, input_folder, output_folder):
		# Where to find images to be identified and where to store them in their 
		# corresponding ID folders
		self.__input_folder = input_folder
		self.__output_folder = output_folder

		print(f"Loading pre-labelled data")

		# Load the list of image filepaths to be labelled
		self.__input_filepaths = DataUtils.allFilesAtDirWithExt(self.__input_folder, ".jpg")

		# There might just be nothing there
		if len(self.__input_filepaths) == 0:
			print(f"No images at: {self.__input_folder}")
			print("Exiting...")
			sys.exit(0)

		# Total number of images to be classified
		self.__total_count = len(self.__input_filepaths)

		# Where images are kept if the user is unsure
		self.__unsure_folder = os.path.join(self.__output_folder, "unsure")

		# Load the dataset that has already been labelled so far (potentially nothing)
		self.__dataset = DataUtils.readFolderDatasetFilepathList(self.__output_folder)

		# Remove the unsure folder
		self.__dataset.pop('unsure', None)

		# A dictionary storing which the index of which image is being currently being displayed
		# as a button image for each ID
		self.__current_disp = {}

		# Initialise the ID counter from this (assumes that IDs have been created
		# properly from 0)
		self.__ID_ctr = len(self.__dataset.keys())

		# The amount of zeros to fill for IDs (e.g. 000, 001, 002) so that they're
		# ordered correctly when considering strings
		self.__zfill_ID = 3

	"""
	Public methods
	"""

	# Called when a query image has been successfully ID'd
	def positiveID(self, ID):
		# Get the current filepath
		query_filepath = self.getQueryImage(filepath_only=True)

		# Strip it down to the filename
		filename = os.path.basename(query_filepath)

		# Construct the new filepath
		folder_dir = os.path.join(self.__output_folder, ID)
		output_filepath = os.path.join(folder_dir, filename)

		# Move the actual file across
		os.rename(query_filepath, output_filepath)

		# Report some info
		print(f"Added image: {filename} to class: {ID}")

		# Remove the head of the filepaths list
		self.__input_filepaths.pop(0)

		# Add the newly labelled file into the dataset list
		self.__dataset[ID].append(output_filepath)

	# Called when a new ID has been requested to be created
	def createNewID(self):
		# Create the full folder directory
		ID_name = str(self.__ID_ctr).zfill(self.__zfill_ID)
		folder_dir = os.path.join(self.__output_folder, ID_name)

		# Make sure the directory doesn't already exist
		assert not os.path.exists(folder_dir)

		# Make the folder
		os.mkdir(folder_dir)

		# Add a new entry in the dataset
		self.__dataset[ID_name] = []

		# Move the first image across
		self.positiveID(ID_name)

		# Report some info
		print(f"Created new ID: {ID_name}")

		# Increment the identity counter
		self.__ID_ctr += 1

		# Return the newly created ID
		return ID_name

	# Create a folder called unsure (if it doesn't exist already), and place the current image in there
	def unsure(self):
		# Create the unsure folder (if it doesn't exist already)
		if not os.path.exists(self.__unsure_folder): os.mkdir(self.__unsure_folder)
		
		# Get the current filepath
		query_filepath = self.getQueryImage(filepath_only=True)

		# Strip it down to the filename
		filename = os.path.basename(query_filepath)

		# Construct the new filepath
		output_filepath = os.path.join(self.__unsure_folder, filename)

		# Move the actual file across
		os.rename(query_filepath, output_filepath)

		# Report some info to the console
		print(f"Reported image: {filename} as user unsure")

		# Remove the head of the filepaths list
		self.__input_filepaths.pop(0)

	"""
	(Effectively) private methods
	"""



	"""
	Getters
	"""

	# Get a new example for a specific class, either a previous or next entry
	def getNewExample(self, ID, increment):
		# Get the index of the image currently being displayed as a button
		current_idx = self.__current_disp[ID]

		# Get the number of current instances for this ID
		num_instances = len(self.__dataset[ID])

		# Compute the new index we're fetching
		new_idx = current_idx + increment

		# Wrap indices round if necessary
		if new_idx >= num_instances: new_idx = 0
		if new_idx < 0: new_idx = num_instances - 1

		# Save the new index
		self.__current_disp[ID] = new_idx

		# Get the file path with this new index
		new_filepath = self.__dataset[ID][new_idx]

		# print(f"new image = {new_filepath}")

		# Load the image into memory and convert colour
		np_image = cv2.cvtColor(cv2.imread(new_filepath), cv2.COLOR_BGR2RGB)

		return np_image

	# Get example images of each class, picks the first landscape image for each class and returns the total
	# number of examples for that class
	def getClassExamples(self):
		examples = {}
		for class_ID, image_filepaths in sorted(self.__dataset.items()):
			# Dict for this class
			example = {}

			# Get the number of images for this class currently
			example['total'] = len(image_filepaths)

			# Assign the first image for the time being
			example['img'] = cv2.cvtColor(cv2.imread(image_filepaths[0]), cv2.COLOR_BGR2RGB)

			# Load a landscape image from the list
			i = 0
			for path in image_filepaths:
				# Store the index we're at
				self.__current_disp[class_ID] = i

				# Load the image into memory and convert colour
				np_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

				# If it's 'landscape', we're done
				if np_image.shape[1] > np_image.shape[0]:
					examples[class_ID] = np_image
					break
				i += 1

			# Add this example to the overall dict
			examples[class_ID] = example

		return examples

	# Called when a new query image is requested to be labelled
	def getQueryImage(self, filepath_only=False):
		if len(self.__input_filepaths) > 0:
			if filepath_only: return self.__input_filepaths[0]
			else: return cv2.imread(self.__input_filepaths[0])
		else: None

	# Get our numerical progress in labelling
	def getProgress(self):
		return self.__total_count - len(self.__input_filepaths), self.__total_count

	def getDataset(self):
		return self.__dataset
	def getTotalCount(self):
		return len(self.__input_filepaths)