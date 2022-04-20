#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("/home/will/work/1-RA/src")
import cv2
from tqdm import tqdm

# My libraries
from Utilities.DataUtils import DataUtils
from Detector.DetectionManager import DetectionManager
from Tracker.TrackingManager import TrackingManager
from Visualiser.VisualisationManager import VisualisationManager as VM

"""
Class for annotating the detection of tracked individuals in video feeds and instances in images
Does NOT do any identification
"""

class DetectionLabeller(object):
	# Class constructor
	def __init__(self, ):
		"""
		Class attributes
		"""

		"""
		Class objects
		"""

		# Species-wide detector
		self.__detector = DetectionManager()

		"""
		Class setup
		"""

		# Setup a named labelling window with opencv
		self.__labelling_window_name = "Will\'s incredible labelling program"
		cv2.namedWindow(self.__labelling_window_name)

	"""
	Public methods
	"""

	# For each individual image in a folder, label them
	def labelImagesInFolder(self, folder):
		# Gather all the images to be labelled
		images = DataUtils.allFilesAtDirWithExt(folder, ".jpg")

		# And let's label them
		pbar = tqdm(total=len(images))
		for image_fp in images:
			print(f"Labelling image at path: {image_fp}")
			self.__labelSingleImage(image_fp)
			pbar.update()
		pbar.close()

	# For eah individual video in a folder, label them
	def labelVideosInFolder(self, rgb_folder, depth_folder):
		# Gather all the videos to be labelled
		rgb_video_fps = DataUtils.allFilesAtDirWithExt(folder, ".mov")
		depth_video_fps = DataUtils.allFilesAtDirWithExt(folder, ".mov")

		# Make sure they're equally sized
		assert len(rgb_video_fps) == len(depth_video_fps)

		# Initialise a tracker
		self.__tracker = TrackingManager()

		# Iterate over each video and label it
		for rgb_path, depth_path in zip(tqdm(rgb_video_fps), depth_video_fps):
			self.__labelSingleVideo(rgb_path, depth_path)

	"""
	(Effectively) private methods
	"""

	def __labelSingleImage(self, image_filepath):
		# Load the image
		image = cv2.imread(image_filepath)

		# Detect on the image
		detections = self.__detector.detect(image)

		# Check these with the operator
		if not self.__checkDetectionsWithUser(image, detections):
			detections = self.__manuallyLabelBoundingBoxes()

		# Not let's label the identities
		annotations = self.__identifyBoundingBoxes(detections)

		# And finally save the annotation to file
		self.__saveAnnotations(annotations)

	def __labelSingleVideo(self, video_filepath):
		# Open the video
		video = cv2.VideoCapture(video_filepath)

		# Get the number of frames
		num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

		# Get the first frame from the video
		_, frame = video.read()

		# Detect on the first frame
		detections = self.__detector.detect(frame)

		# Check these with the operator
		if not self.__checkDetectionsWithUser(image, detections):
			detections = self.__manuallyLabelBoundingBoxes()

		# Initialise our tracker(s) with this
		self.__tracker.reset()
		self.__tracker.initialise(frame, detections)

		# List of detections over frames
		tracked_detections = []

		# Iterate over each frame
		pbar = tqdm(total=num_frames)
		while video.isOpened():
			# Get a new frame
			_, frame = video.read()

			# Update the tracker(s)
			tracked = self.__tracker.update(frame)

			# Save it
			tracked_detections.append(tracked)

			# Update on progress
			pbar.update()
		pbar.close()

		# Save the 

	# Check whether the usre is happy with the detections being the ground truth bounding box
	def __checkDetectionsWithUser(self, image, detections): 
		# Visualise the detections
		disp_img, _ = VM.plotDetections(image, detections, display=False, convert_dets=False)

		# Is the user happy with the detections being the ground truth bounding box
		satisfied = False
		cv2.imshow("Box labelling", disp_img)
		print("Are you happy with the detections? (y/n): ")
		while True:
			c = cv2.waitKey(0)
			if c == ord("y"): satisfied = True; break
			if c == ord("n"): break
			else: print(f"Invalid answer, try again.")

		return satisfied

	# For a given image, let's manually label the bounding boxes
	def __manuallyLabelBoundingBoxes(self, image):
		return self.__bbox_gui.labelImage(image, )

	"""
	Getters
	"""

	"""
	Setters
	"""

	"""
	Static methods
	"""


# Entry method/unit testing method
if __name__ == '__main__':
	# Let's label a folder full of images
	labeller = Labeller()
	folder = "/home/will/work/1-RA/src/Detector/data/consolidated_augmented"
	labeller.labelImagesInFolder(folder)

	# Let's label a folder full of videos
	# labeller = Labeller()
	# folder = "foo/bar/videos"
	# labeller.labelVideosInFolder(folder)
