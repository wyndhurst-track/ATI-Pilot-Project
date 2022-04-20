# Core libraries
import os
import sys
sys.path.append(os.path.abspath("../"))
import cv2
import json
import math
import random
import numpy as np
# import seaborn as sns
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from mpl_toolkits.axes_grid1 import ImageGrid

# My libraries
from Utilities.DataUtils import DataUtils
from Datasets.RGBDCows2020 import RGBDCows2020

class DatasetVisualiser(object):
	"""
	Class for visualising stats/graphs regarding a dataset
	"""

	def __init__(self):
		"""
		Class constructor
		"""

		pass

	"""
	Public methods
	"""

	"""
	(Effectively) private methods
	"""

	"""
	Staticmethods
	"""

	@staticmethod
	def binsLabels(bins, **kwargs):
		bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
		plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
		plt.xlim(bins[0], bins[-1])

	@staticmethod
	def tileCategories(dataset, max_cols=20, img_type="RGB", random=True):
		""" Produce a tiled image of all categories

		Retrieve a random instance from each category and display them all in a grid of tiled images.
		"""

		# Load up the file
		with open(dataset.getSplitsFilepath()) as handle:
			random_splits = json.load(handle)
			print(f"Retrieved splits at {dataset.getSplitsFilepath()}")

		# Which fold shall we use
		if random: fold = random.choice(list(random_splits.keys()))
		else: fold = str(0)

		# Retrieve a random example for each category
		if random: examples = {cat: random.choice(random_splits[fold][cat]['train']) for cat in random_splits[fold].keys()}
		else: examples = {cat: random_splits[fold][cat]['train'][0] for cat in random_splits[fold].keys()}

		# Get the root directory for the dataset
		root_dir = os.path.join(dataset.getDatasetPath(), img_type)

		# Actually load the images into memory
		example_images = {cat: cv2.imread(os.path.join(root_dir, cat, filename)) for cat, filename in examples.items()}

		# Retrieve the number of classes we have
		num_classes = dataset.getNumClasses()
		assert num_classes == len(example_images.keys())

		# Compute the number of rows needed
		nrows = int(math.ceil(num_classes/max_cols))

		# Create the figure
		fig = plt.figure(figsize=(40,10))
		grid = ImageGrid(fig, 111, nrows_ncols=(nrows, max_cols), axes_pad=0.1)

		# Go through and turn each axes visibility off
		for ax in grid:
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.axis('off')

		# Iterate through each category in order
		for ax, cat in zip(grid, sorted(example_images.keys())):
			# Get the numpy image
			np_img = example_images[cat]

			# Convert to RGB (from BGR)
			np_img = np_img[...,::-1].copy()

			# Convert to PIL
			pil_img = Image.fromarray(np_img, "RGB")

			# Assign the image to this axis
			ax.imshow(pil_img)

		plt.tight_layout()
		# plt.show()
		plt.savefig(f"{img_type}_category-examples.png")

	@staticmethod
	def visualiseObjectSizeDistribution(anno_path):
		"""

		"""

		# Dataset ranges
		outdoor1 = [2287, 3707]	# PhD 18 cows
		indoor0 = [0, 939]
		indoor1 = [1040, 2286]
		outdoor0 = [940, 1039]
		
		# Get all the annotation files
		anno_filepaths = DataUtils.allFilesAtDirWithExt(anno_path, ".xml")

		# List of object sizes
		obj_sizes = []

		# List of object counts per image
		obj_counts = []

		# Iterate through each annotation file
		for anno_fp in tqdm(anno_filepaths):
			# Load it into memory
			annotation = DataUtils.readXMLAnnotation(anno_fp)

			# Count the number of annotations/objects
			obj_counts.append(len(annotation['objects']))

			# Iterate through each object
			for obj in annotation['objects']:
				# Find the area of the annotation
				w = obj['x2'] - obj['x1']
				h = obj['y2'] - obj['y1']

				# Add it to the list
				obj_sizes.append({'filenumber': int(os.path.basename(anno_fp)[:-4]), 'size': w * h})

		outdoor1_sizes = []
		indoor_sizes = []
		outdoor0_sizes = []
		
		# Split the object sizes into constituent components
		for size in obj_sizes:
			# Extract the filename int
			fn = size['filenumber']

			# Which category does it belong in
			if fn >= indoor0[0] and fn <= indoor0[1]: indoor_sizes.append(size['size'])
			elif fn >= outdoor0[0] and fn <= outdoor0[1]: outdoor0_sizes.append(size['size'])
			elif fn >= indoor1[0] and fn <= indoor1[1]: indoor_sizes.append(size['size'])
			elif fn >= outdoor1[0] and fn <= outdoor1[1]: outdoor1_sizes.append(size['size'])

		alpha_val = 0.5

		# For object sizes
		plt.figure()
		plt.hist(outdoor1_sizes, 100, alpha=alpha_val, label='(a)')
		plt.hist(indoor_sizes, 100, alpha=alpha_val, label='(b)')
		plt.hist(outdoor0_sizes, 100, alpha=1, label='(c)')
		plt.legend(loc='upper right')
		plt.xlim((0, 400000))
		plt.xlabel("Pixel Area")
		plt.ylabel("Occurences")
		plt.tight_layout()
		# plt.show()
		plt.savefig("object-size-dist.pdf")

		print(f"Total object count =  {np.sum(np.array(obj_counts))}")

		# For object count per image
		# plt.figure()
		# bins = range(10)
		# plt.hist(obj_counts, bins=bins)
		# DatasetVisualiser.binsLabels(bins, fontsize=10)
		# plt.xlim((1,9))
		# plt.xlabel("Objects Per Image")
		# plt.ylabel("Occurences")
		# plt.tight_layout()
		# # plt.show()
		# plt.savefig("object-count.pdf")

	@staticmethod
	def visualiseInstancesFolderDataset(splits_dir):
		""" Visualise the instances per category

		Similarly to `visualiseInstancesPerDay`, generate a bar chart

		Arguments:
			train_path: `str`. Full file path to the training folder dataset
			test_path: `str`. Full file path to the testing folder dataset
		"""

		# Load the splits file
		with open(splits_dir) as handle:
			splits = json.load(handle)

		# Get an ordered list of counts
		train_list = [len(splits[x]['train']) for x in sorted(splits.keys())]
		valid_list = [len(splits[x]['valid']) for x in sorted(splits.keys())]
		test_list = [len(splits[x]['test']) for x in sorted(splits.keys())]


		# Get the number of classes
		num_classes = len(splits.keys())

		# Create the figure
		plt.figure(figsize=(10,5))

		# Width of the bars
		width = 0.85

		# X indices
		ind = np.arange(num_classes) + 1

		ax = plt.axes()
		fs = 14.0
		a_x = 0.192
		b_x = 0.606
		c_x = 0.915
		ax.annotate('(a)', xy=(a_x, 0.97), xytext=(a_x, 1.05), xycoords='axes fraction', 
			fontsize=fs*1.5, ha='center', va='bottom',
			bbox=dict(boxstyle='square', fc='white'),
			arrowprops=dict(arrowstyle='-[, widthB=5.6, lengthB=5.5', lw=2.0))
		ax.annotate('(b)', xy=(b_x, 0.97), xytext=(b_x, 1.05), xycoords='axes fraction', 
			fontsize=fs*1.5, ha='center', va='bottom',
			bbox=dict(boxstyle='square', fc='white'),
			arrowprops=dict(arrowstyle='-[, widthB=7.2, lengthB=9', lw=2.0))
		ax.annotate('(c)', xy=(c_x, 0.97), xytext=(c_x, 1.05), xycoords='axes fraction', 
			fontsize=fs*1.5, ha='center', va='bottom',
			bbox=dict(boxstyle='square', fc='white'),
			arrowprops=dict(arrowstyle='-[, widthB=2.3, lengthB=9', lw=2.0))

		# Create the bar plot
		p1 = plt.bar(ind, train_list, width)
		p2 = plt.bar(ind, valid_list, width, bottom=train_list)
		bottom = np.array(valid_list) + np.array(train_list)
		p3 = plt.bar(ind, test_list, width, bottom=bottom.tolist())

		plt.xlim((0, num_classes+1))
		plt.xticks(ind, tuple(str(x).zfill(2) for x in ind.tolist()), rotation="vertical")
		plt.legend((p1[0], p2[0], p3[0]), ('Train', 'Valid', 'Test'), loc="center right")
		plt.xlabel("Class (Individual Animals)")
		plt.ylabel("Instances")
		plt.tight_layout()
		# plt.show()
		plt.savefig("instance-dist.pdf")

	@staticmethod
	def visualiseCowNumbers():
		""" Visualise the distribution of cow breeds per country for the CEiA paper

		Source: http://www.whff.info/documentation/statistics.php#go1
		"""

		countries = [	"United States", "New Zealand", "Germany", "France", "Poland",
						"Italy", "UK", "Ireland", "Australia", "Canada", "Japan",
						"Spain", "Denmark", "Switzerland", "Austria", "Czechia",
						"Sweden", "Finland", "Hungary", "Croatia", "Slovakia",
						"Latvia"	]

		non_holstein = [1034000, 3325281, 1809569, 1200000, 332114, 450000,
						125000, 302833, 468300, 68061, 8500, 22000, 166931,
						380000, 420000, 145000, 160000, 237390, 10000,
						100332, 59431, 77717]

		holstein = [	8366000, 1667633, 2291331, 2500000, 1881978, 1450000,
						1758000, 1500000, 1092700, 904239, 838700, 795000,
						396869, 140000, 80000, 220000, 140000, 29500, 240000,
						36215, 68440, 45283]

		ind = np.arange(len(holstein))

		width = 0.85
		# sf = 1000000

		# Create the figure
		plt.figure(figsize=(10,5))
		ax = plt.subplot(111)

		p1 = plt.bar(ind, np.array(holstein), width)
		p2 = plt.bar(ind, np.array(non_holstein), width, bottom=holstein)

		plt.xticks(ind, countries, rotation=90)
		ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))

		plt.legend((p1[0], p2[0]), ('Holstein', 'Non-Holstein'))
		plt.ylabel("Number of dairy cows")
		plt.tight_layout()
		plt.savefig("country-dist.pdf")
		# plt.show()

	@staticmethod
	def visualiseInstancesPerDay(splits_filepath, num_classes):
		""" Visualise instances per category per day

		For a .json splits file with instances separated by class by day, plot the distribution
		of instances across each class across each day and save the figure to file

		Arguments:
			splits_filepath: `str`. Full filepath to the splits .json file.
			num_classes: `int`. The number of unique classes/categories for this dataset.
		"""

		# Load up the file
		with open(splits_filepath) as handle:
			day_splits = json.load(handle)

		# X indices
		ind = np.arange(num_classes)

		# Create the figure
		plt.figure(figsize=(33,13))

		# Width of the bars
		width = 0.85

		# Go through and create a dictionary of number of instances per class per day
		instances = {day: [len(v[x]) for x in sorted(v.keys())] for day, v in day_splits.items()}

		# Create a unique colour palette
		palette = np.array(sns.color_palette("hls", len(instances.keys())))

		# Maintain a list of the bars
		bars = []

		# Keep a sum of the previous bars
		bar_sum = np.zeros(num_classes)

		# Iterate through each day
		for i, day in enumerate(sorted(instances.keys())):
			# Get the current ordered list of category lengths
			category_lengths = instances[day]

			# Create the bar plot
			bar = plt.bar(ind, category_lengths, width, bottom=bar_sum.tolist(), color=palette[i])

			# Add it to the list of bars
			bars.append(bar)

			# Total up the bars for the next one
			bar_sum += category_lengths

		print(f"Instances: mean, stddev, min, max")
		print(np.mean(bar_sum), np.std(bar_sum), np.min(bar_sum), np.max(bar_sum))

		plt.xlim((-1, num_classes))
		plt.ylim((0, np.max(bar_sum)+10))
		plt.xticks(ind, tuple(str(x).zfill(3) for x in ind.tolist()), rotation="vertical")
		plt.xlabel("Class #")
		plt.ylabel("Instances")
		plt.legend(tuple(b[0] for b in bars), tuple(d for d in sorted(instances.keys())), loc="upper right")
		plt.tight_layout()
		# plt.show()
		plt.savefig("RGBDCows2020-instance-dist.pdf")

# Entry method/unit testing method
if __name__ == '__main__':
	# Visualise examples of the dataset
	dataset = RGBDCows2020(suppress_info=True)
	DatasetVisualiser.tileCategories(dataset, img_type="Depth", random=False)

	# Visualise the distribution of instances per day for the RGBDCows2020 dataset
	# dataset = RGBDCows2020(split_mode="day", num_testing_days=3, suppress_info=True)
	# splits_filepath = dataset.getSplitsFilepath()
	# num_classes = dataset.getNumClasses()
	# DatasetVisualiser.visualiseInstancesPerDay(splits_filepath, num_classes)

	# Visualise the distribution of instances across the OpenCows2020 dataset
	# splits_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\OpenCows2020\\identification\\images\\train_valid_test_splits.json"
	# DatasetVisualiser.visualiseInstancesFolderDataset(splits_dir)

	# Visualise the distribution of object sizes
	# anno_path = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\labels-xml"
	# DatasetVisualiser.visualiseObjectSizeDistribution(anno_path)

	# Visualise the distribution of cow breeds per country
	# DatasetVisualiser.visualiseCowNumbers()