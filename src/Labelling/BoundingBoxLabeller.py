#!/usr/bin/env python

import cv2
import sys
import numpy as np
import BoundingBoxGUIHandler
import xml.etree.ElementTree as ET
from xml.dom import minidom
import glob
import os.path
from tqdm import tqdm

"""
This class is for annotating bounding boxes for objects in images
These can then be passed on for:
	(a) training object detectors or
	(b) identification components
"""

class BoundingBoxLabeller:
	# Class constructor
	def __init__(self):
		# String handler for window name
		self._labelling_window_name = 'Will\'s incredible labelling program'

		# Width of labelling window
		self._labelling_window_width = 1400

		self._bbox_gui = BoundingBoxGUIHandler.BoundingBoxGUIHandler(self._labelling_window_name, self._labelling_window_width)

	# Stores an annotation into the VOC/caffe XML format
	def storeBoundingBoxesToFile(self, bboxes, image, image_name, save_path):
		root = ET.Element("annotation")

		folder = ET.SubElement(root, "folder")
		filename = ET.SubElement(root, "filename").text = str(image_name)

		i_height, i_width, i_channels = image.shape

		size = ET.SubElement(root, "size")
		width = ET.SubElement(size, "width").text = str(i_width)
		height = ET.SubElement(size, "height").text = str(i_height)
		depth = ET.SubElement(size, "depth").text = str(i_channels)

		for box in bboxes:
			object_e = ET.SubElement(root, "object")
			name = ET.SubElement(object_e, "name").text = "TO BE LABELLED"

			bndbox= ET.SubElement(object_e, "bndbox")

			xmin = box[0][0]
			xmax = box[1][0]
			ymin = box[0][1]
			ymax = box[1][1]

			# Swap values if necessary
			if xmin > xmax:
				xmin, xmax = xmax, xmin
			if ymin > ymax:
				ymin, ymax = ymax, ymin

			xmin_e = ET.SubElement(bndbox, "xmin").text = str(xmin)
			xmax_e = ET.SubElement(bndbox, "xmax").text = str(xmax)
			ymin_e = ET.SubElement(bndbox, "ymin").text = str(ymin)
			ymax_e = ET.SubElement(bndbox, "ymax").text = str(ymax)

		# Prettify the xml
		rough_string = ET.tostring(root, 'utf-8')
		reparsed = minidom.parseString(rough_string)
		tree = reparsed.toprettyxml(indent="    ")

		# And save it to file
		with open(save_path, "w") as text_file:
			text_file.write(tree)

	# Label objects on a folder of images
	def labelBoundingBoxes(self, input_path, output_path):
		cv2.namedWindow(self._labelling_window_name)

		print 'Let\'s label some cow bounding boxes, whooo\n'
		print 'LABELLING GUIDELINES:\n'
		print 'Label all cows unless:'
		print '\t-You are unsure whether or not it is a cow'
		print '\t-The object is very small (at your discretion'
		print '\t-Less than 10-20% of the cow is visible'
		print 'Bounding boxes:'
		print '\t-Mark the visible area of the cow (not the estimated area)'
		print '\t-Should contain all visible pixels of the cow but don\'t include the tail'
		print 'Occlusion/truncation:'
		print '\tIf more than 15-20% of the cow is occluded, do NOT label it\n'

		# Collect images
		image_files = [os.path.join(input_path, x) for x in sorted(os.listdir(input_path)) if x.endswith("jpg")]

		# Main labelling loop!
		pbar = tqdm(total=len(image_files))
		for current_image in image_files:
			image_name = current_image.split("/")[-1]
			image_name = image_name[0: image_name.find(".")]

			save_path = output_path + image_name + ".xml"

			# Annotation file already exists, skip
			if os.path.isfile(save_path):
				print 'File {:s} has already been labelled, skipping...'.format(image_name)
			# File hasn't been labelled, label it
			else:
				image = cv2.imread(current_image)

				bboxes = self._bbox_gui.labelImage(image, image_name)

				self.storeBoundingBoxesToFile(bboxes, image, image_name + ".jpg", save_path)
				print bboxes
				print "\n"

			pbar.update()
		pbar.close()

# Entry method
if __name__ == '__main__':
	bbox_labeller = BoundingBoxLabeller()

	input_path = "/home/will/work/1-RA/src/Utilities/BoundingBoxLabeller/data/images/"
	output_path = "/home/will/work/1-RA/src/Utilities/BoundingBoxLabeller/data/output/"

	bbox_labeller.labelBoundingBoxes(input_path, output_path)
