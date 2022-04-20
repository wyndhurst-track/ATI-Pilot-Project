#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import cv2
import csv
import datetime
import numpy as np
from tqdm import tqdm

# My libraries
from config import cfg
from Utilities.Utility import Utility
from Utilities.DataUtils import DataUtils
from Utilities.ImageUtils import ImageUtils

"""
Descriptor

use matplotlib to visualise cow + 2D tracking position? and camera coverage?
"""

class VisualisationManager(object):
	# Class constructor
	def __init__(self):
		"""
		Class attributes
		"""

		pass

		"""
		Class objects
		"""

		"""
		Class setup
		"""

	"""
	Public methods
	"""	

	"""
	(Effectively) private methods
	"""

	"""
	Getters
	"""

	"""
	Setters
	"""

	"""
	Static methods
	"""

	# Mostly for identification labelling, draw a matrix of tiled images and their ID,
	# for the provided dataset, it chooses the first image in the list
	@staticmethod
	def drawIdentificationTiles(dataset, display=False):
		# Constant definitions
		img_w = 1280
		img_h = 720
		tile_size_w = 100
		tile_gap_hor = 10
		tile_gap_ver = 50

		# Create the render image and fill it with white
		render_img = np.zeros((img_h, img_w, 3), np.uint8)
		render_img.fill(255)

		# The number of classes
		num_classes = len(dataset.keys())

		# Loop over each class and render them
		row = 0
		col = 0
		for class_id, images in sorted(dataset.items()):
			# Compute the current x and y for the image
			x = (col * tile_size_w) + ((col+1) * tile_gap_hor)
			y = (row * tile_size_w) + ((row+1) * tile_gap_ver)

			# Check whether we should go to a new line, recompute x,y if needed
			if x + tile_size_w > img_w:
				row += 1
				col = 0
				x = tile_gap_hor
				y = (row * tile_size_w) + ((row+1) * tile_gap_ver)

			# Get a landscape image from the list
			for tile_image in images: 
				if tile_image.shape[1] > tile_image.shape[0]: 
					break

			# Resize it to the size we're after
			scale_factor = tile_size_w / float(tile_image.shape[1])
			calculated_h = int(scale_factor * tile_image.shape[0])
			if calculated_h > tile_size_w: calculated_h = tile_size_w
			tile_image = cv2.resize(tile_image, (tile_size_w, calculated_h))
			tile_size_h = tile_image.shape[0]

			# Insert the tile image into the larger image
			render_img[y:y+tile_size_h, x:x+tile_size_w, :] = tile_image[:]

			# Insert the class ID as text above the tiled image
			cv2.putText(render_img, str(class_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

			# Increment any counters
			col += 1

		if display:
			cv2.imshow("Tiled image", render_img)
			cv2.waitKey(0)

		return render_img

	# Plot detections on an image, returns the rendered image
	# Can also plot ground truth boxes on top
	# If TP_only=True, then it only plots detections where the IoU with some GT was greater
	# than a provided threshold
	@staticmethod
	def plotDetections(	img, 
						detections, 
						display=True, 
						GT_boxes=[], 
						TP_only=False, 
						IoU_thresh=cfg.DET.IoU_THRESHOLD, 
						thickness=2, 
						convert_GTs=False,
						convert_dets=False,
						save_failures=False					):
		# Define the render image
		render_img = img.copy()

		# Get the dimensions we're working with
		img_w = render_img.shape[1]
		img_h = render_img.shape[0]

		# If we're supposed to, convert the detections from (centrepoint, w, h) to (x1,y1,x2,y2)
		bboxes = []
		if detections is not None:
			if convert_dets:
				for box in detections:
					x1 = int(box[0] - box[2]/2)
					y1 = int(box[1] - box[3]/2)
					x2 = x1 + int(box[2])
					y2 = y1 + int(box[3])
					bboxes.append([x1,y1,x2,y2])
			else:
				for d in detections['objects']:
					x1 = d['x1']
					y1 = d['y1']
					x2 = d['x2']
					y2 = d['y2']
					bboxes.append([x1,y1,x2,y2])

		# We might need to convert from darknet to pixel
		# Convert ground truthes from decimal [0,1] (x,y,w,h) to pixel (x1,y1,x2,y2)
		GT_bboxes = []
		if convert_GTs:
			for gt in GT_boxes:
				x1 = int((gt[0] - gt[2]/2) * img_w)
				y1 = int((gt[1] - gt[3]/2) * img_h)
				x2 = x1 + int(gt[2] * img_w)
				y2 = y1 + int(gt[3] * img_h)
				GT_bboxes.append([x1,y1,x2,y2])
		# Just extract from the dict
		else:
			for gt in GT_boxes['objects']:
				x1 = gt['x1']
				y1 = gt['y1']
				x2 = gt['x2']
				y2 = gt['y2']
				GT_bboxes.append([x1,y1,x2,y2])

		# Only plot TP detections in blue by looking at IoU between detections and GT
		if TP_only:
			# Construct the list of TP detections
			TPs = [bbox for bbox in bboxes for gt in GT_bboxes if Utility.bboxIntersectionOverUnion(bbox, gt) >= IoU_thresh]

			# Render TP detections in BLUE
			for TP in TPs: cv2.rectangle(render_img, (TP[0], TP[1]), (TP[2], TP[3]), (255,0,0), thickness)

			# Compute precision
			precision = float(len(TPs)) / float(len(GT_bboxes))

			# Precision may exceed one if there are multiple detections per gt box
			# is this a problem?
			if precision > 1: precision = 1.0
			# If we're supposed to save failure cases to file, do so!
			elif save_failures and precision < 1:
				# Copy the main image
				failure_img = img.copy()

				# Freshly render the detections and ground truth
				for bbox in bboxes: cv2.rectangle(failure_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), thickness)
				for gt in GT_bboxes: cv2.rectangle(failure_img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,0), thickness)

				# Generate a filepath and save the image with detections/GT rendered
				failure_path = "/home/will/work/1-RA/src/Detector/data/failure_cases"
				filepath = os.path.join(failure_path, str(datetime.datetime.now())+".jpg")
				cv2.imwrite(filepath, failure_img)
				print(f"Saved failure case to: {filepath}")
		else:
			# Render detections in RED
			if detections is not None:
				for bbox in bboxes: cv2.rectangle(render_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), thickness)

			# Render ground truthes in GREEN
			for gt in GT_bboxes: cv2.rectangle(render_img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,0), thickness)

			# Precision needs defining
			precision = -1

		# Display the image if we're supposed to
		if display:
			# Show the rendered image
			cv2.imshow("Detections", render_img)
			cv2.waitKey(0)

		return render_img, precision

	@staticmethod
	def drawRotatedBbox(img, annotations, display=False, display_head=False):
		""" Given an image, render a set of rotated annotations """

		# Make a copy
		render_img = img.copy()

		# Loop over every object in this annotation
		for obj in annotations:
			# Get pixel coordinates of each corner
			((x1,y1),(x2,y2),(x3,y3),(x4,y4)) = ImageUtils.rotatedRectToPixels(obj)

			# Extract the annotation centrepoint
			cx = int(obj['cx'])
			cy = int(obj['cy'])

			# Render the box, centre, head direction and top left box corner
			cv2.circle(render_img, (cx, cy), 5, (255,0,0), 5)

			# Is there a Identity included, render it if so
			if "ID" in obj.keys():
				ID = str(obj['ID']).zfill(3)
				cv2.putText(render_img, f"ID={ID}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

			# Head direction
			if display_head:
				cv2.circle(render_img, (int(obj['head_cx']), int(obj['head_cy'])), 5, (0,0,255), 5)

			# Top of box (and therefore head direction)
			top_x = int((x1+x2)/2)
			top_y = int((y1+y2)/2)
			cv2.circle(render_img, (top_x, top_y), 5, (0,255,0),5)

			# The box
			cv2.line(render_img, (x1,y1), (x2,y2), (255,0,0), 2)
			cv2.line(render_img, (x2,y2), (x4,y4), (255,0,0), 2)
			cv2.line(render_img, (x4,y4), (x3,y3), (255,0,0), 2)
			cv2.line(render_img, (x3,y3), (x1,y1), (255,0,0), 2)

		# Display if we're supposed to
		if display:
			cv2.imshow("Visualisation", render_img)
			cv2.waitKey(0)

		return render_img

	@staticmethod
	def visualiseFromCSV(path_to_csv=None):
		""" From a CSV file originating from processed """

		# Sort out pathing to the CSV file, if there isn't one specified, try and find it at the normal location
		if path_to_csv is None:
			path_to_csv = os.path.join(cfg.DAT.CSV_PATH, cfg.DAT.CSV_FILE)

		# Dictionary of video files to open individually and visualise from
		video_filenames = {}

		# Open the file
		with open(path_to_csv, newline='') as handle:
			# Create the reader
			reader = csv.reader(handle, delimiter=";", quotechar='"')

			# Iterate through each row
			for i, row in tqdm(enumerate(reader), desc="Processing CSV file"):
				# Skip the first row as it just has column headings
				if i > 0:
					# Extract the video filename for this observation
					filename = row[-1]

					# Initialise the list for this filename if it is the first observation
					if filename not in video_filenames.keys():
						video_filenames[filename] = [row]

					# Add it to the list
					else:
						video_filenames[filename].append(row)

		# Now iterate through each video we've extracted
		for filename in sorted(video_filenames.keys()):
			# Find the full video file in the processed folder in the workspace
			video_filepath = os.path.join(cfg.FSM.WORKSPACE, cfg.FSM.FOLDER_4, filename)
			print(f"Attempting to visualise video file at path: {video_filepath}")

			# If it doesn't exist, progress to the next file
			if not os.path.exists(video_filepath):
				print(f"Couldn't find file at path: {video_filepath}")
				print(f"Continuing to next video file")
				continue

			# Open the video
			video = cv2.VideoCapture(video_filepath)

			# Keep count of the frame number
			frame_ctr = 0

			# Iterate through it
			while video.isOpened():
				# Get a frame
				ret, frame = video.read()

				# Increment the counter
				frame_ctr += 1

				# Ensure it isn't an empty frame
				if ret:
					# List of objects to be visualised
					tbv = []

					# Iterate through the list of observations for this video file
					for obvs in video_filenames[filename]:
						# Does the frame ID match the current frame
						if int(obvs[-2]) == frame_ctr:
							# Let's create an object dictionary
							obj = {}
							obj['ID'] = int(obvs[2])
							obj['cx'] = float(obvs[3])
							obj['cy'] = float(obvs[4])
							obj['w'] = float(obvs[5])
							obj['h'] = float(obvs[6])
							obj['angle'] = float(obvs[7])

							# Add this dictionary to the list
							tbv.append(obj)

					# Only visualise if the list is populated (otherwise there wasn't a frame ID match)
					if len(tbv) > 0:
						VisualisationManager.drawRotatedBbox(frame, tbv, display=True)

				# We've reached the end of the video, stop
				else:
					break

			print(f"Finished visualising: {filename}")

# Entry method/unit testing method
if __name__ == '__main__':
	# Testing tiled 
	# folder = "/home/will/work/1-RA/src/Datasets/data/CowID-PhD/raw"
	# dataset = DataUtils.readFolderDataset(folder)
	# VisualisationManager.drawIdentificationTiles(dataset, display=True)
	
	# Visualise the output of processing
	VisualisationManager.visualiseFromCSV()