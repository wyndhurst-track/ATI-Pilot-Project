# Core libraries
import os
import cv2
import glob
import yaml
import torch
import random
import numpy as np
from PIL import Image
import scipy.misc as m

# PyTorch
from torch.utils import data

def ordered_glob(rootdir='.', instances=''):
	"""Performs recursive glob with given suffix and rootdir 
		:param rootdir is the root directory
		:param suffix is the suffix to be searched
	"""
	filenames = []

	folders = glob.glob(rootdir + "/*")

	for folder in folders:

		folder_id = os.path.split(folder)[1][0:6]

		if folder_id in instances:

			folder_path = folder + "/*"

			filenames_folder = glob.glob(folder_path)
			filenames_folder.sort()
			filenames.extend(filenames_folder)

	return filenames

class triplet_resnet_open_cows(data.Dataset):

	def __init__(	self, 
					root, 
					split="train", 
					is_transform=False, 
					img_size=(224,224), 
					augmentations=None, 
					instances=None			):
		self.root = root
		self.split = split
		self.is_transform = is_transform
		self.augmentations = augmentations
		self.n_classes = 46
		self.n_channels = 3
		self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
		self.files = {}

		self.images_base = os.path.join(self.root, self.split)

		self.files[split] = ordered_glob(rootdir=self.images_base,  instances=instances)

		self.instances = instances

		if not self.files[split]:
			raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

		print("Found %d %s images" % (len(self.files[split]), split))

	# Overrides superclass
	def __len__(self):
		"""__len__"""
		return len(self.files[self.split])

	# Overrides superclass
	def __getitem__(self, index):
		"""__getitem__

		:param index:
		"""
		img_path = self.files[self.split][index].rstrip()
		img = Image.open(img_path)
		img = self.resize_keepRatio(img)

		# Get the class this index refers to
		current_category = img_path.split("/")[-2]
		assert current_category in self.instances

		# Get the label
		lbl = np.array([int(current_category)])

		# And convert the image to numpy
		img = np.array(img, dtype=np.uint8)

		# Get the filename this index refers to
		current_imagename = os.path.basename(img_path)

		# (POSITIVE) Get another random image from this class
		category_dir = os.path.join(self.images_base, current_category)
		possible_list = [x for x in self.files[self.split] if category_dir in x]
		possible_list.remove(img_path)
		img_path_similar = random.choice(possible_list)

		# (NEGATIVE) Get a random image from a different random class
		possible_categories = list(self.instances)
		possible_categories.remove(current_category)
		random_category = random.choice(possible_categories)
		random_category_path = os.path.join(self.images_base, random_category)
		different_list = [x for x in self.files[self.split] if random_category_path in x]
		img_path_different = random.choice(different_list)

		# Load these images
		img_pos = Image.open(img_path_similar)
		img_neg = Image.open(img_path_different)

		# Get the labels
		lbl_pos = np.array([int(current_category)])
		lbl_neg = np.array([int(random_category)])
		
		# Resize the images
		img_pos = self.resize_keepRatio(img_pos)
		img_neg = self.resize_keepRatio(img_neg)

		# Convert to numpy
		img_pos = np.array(img_pos, dtype=np.uint8)
		img_neg = np.array(img_neg, dtype=np.uint8)

		# self.__visualiseTriplet(img, img_pos, img_neg, lbl)

		# Augment/transform
		if self.is_transform:
			img, img_pos, img_neg = self.transform(img, img_pos, img_neg)
			lbl, lbl_pos, lbl_neg = self.transform_lbl(lbl, lbl_pos, lbl_neg)

		return img, img_pos, img_neg, lbl, lbl_neg

	def __visualiseTriplet(self, image_anchor, image_pos, image_neg, label_anchor):
		print(f"Label={label_anchor}")
		cv2.imshow(f"Label={label_anchor} anchor", image_anchor)
		cv2.imshow(f"Label={label_anchor} positive", image_pos)
		cv2.imshow(f"Label={label_anchor} negative", image_neg)
		cv2.waitKey(0)

	# Transform a sample (anchor, positive, negative)
	#
	# This is where I would implement augmentations etc. https://pytorch.org/docs/stable/torchvision/transforms.html
	# Just make sure not to be doing this on the test data
	def transform(self, img, img_pos, img_neg):
		"""transform

		:param img:
		:param lbl:
		"""
		# img = img[:, :, ::-1]
		# img = img.astype(np.float64)
		# # img -= self.mean
		# img = m.imresize(img, (self.img_size[0], self.img_size[1]))
		# # Resize scales images from 0 to 255, thus we need
		# # to divide by 255.0
		# img = img.astype(float) / 255.0
		# NHWC -> NCWH
		img = img.transpose(2, 0, 1)


		# img_pos = img_pos[:, :, ::-1]
		# img_pos = img_pos.astype(np.float64)
		# # img_pos -= self.mean
		# img_pos = m.imresize(img_pos, (self.img_size[0], self.img_size[1]))
		# # Resize scales images from 0 to 255, thus we need
		# # to divide by 255.0
		# img_pos = img_pos.astype(float) / 255.0
		# NHWC -> NCWH
		img_pos = img_pos.transpose(2, 0, 1)

		# img_neg = img_neg[:, :, ::-1]
		# img_neg = img_neg.astype(np.float64)
		# # img_neg -= self.mean
		# img_neg = m.imresize(img_neg, (self.img_size[0], self.img_size[1]))
		# # Resize scales images from 0 to 255, thus we need
		# # to divide by 255.0
		# img_neg = img_neg.astype(float) / 255.0
		# NHWC -> NCWH
		img_neg = img_neg.transpose(2, 0, 1)


		img = torch.from_numpy(img).float()
		img_pos = torch.from_numpy(img_pos).float()
		img_neg = torch.from_numpy(img_neg).float()

		return img, img_pos, img_neg

	def transform_lbl(self, lbl, lbl_pos, lbl_neg):
		lbl = torch.from_numpy(lbl).long()
		lbl_pos = torch.from_numpy(lbl_pos).long()
		lbl_neg = torch.from_numpy(lbl_neg).long()

		return lbl, lbl_pos, lbl_neg

	def resize_keepRatio(self, img):
		old_size = img.size

		ratio = float(self.img_size[0])/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])

		img = img.resize(new_size, Image.ANTIALIAS)

		new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
		new_im.paste(img, ((self.img_size[0]-new_size[0])//2,
					(self.img_size[1]-new_size[1])//2))

		return new_im

# Main/entry function
if __name__ == '__main__':
	# data_path = "/home/will/work/1-RA/src/Datasets/data/CowID-PhD/split"
	data_path = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/split"

	gt_file = '/home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network/loader/dataset_splits.yml'

	with open(gt_file, 'r') as f:
		doc = yaml.load(f, Loader=yaml.FullLoader)

	dataset = triplet_resnet_open_cows(data_path, instances=doc['open_cows']['known'])

	test = dataset[0]