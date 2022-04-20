#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import cv2
import time
import json
import pickle
import numpy as np
from tqdm import tqdm
#from pydarknet import Detector, Image

# My libraries
from config import cfg
from Utilities.DataUtils import DataUtils
from Utilities.EvaluationUtils import EvaluationUtils
# from Detector.DarknetWrapper import DarknetWrapper
# from Detector.FasterRCNNWrapper import FasterRCNNWrapper
# from Detector.YoloPyTorchWrapper import YoloPyTorchWrapper
from Detector.RotatedRetinaNetWrapper import RotatedRetinaNetWrapper
from Visualiser.VisualisationManager import VisualisationManager as VM

"""
This class manages everything to do with the Detection and Localisation of cattle via YOLOv3
or alternative method
"""

class DetectionManager():
	# Class constructor
	def __init__(self, load_weights=True):
		"""
		Class attributes
		"""

		# The size of the batch when processing multiple images
		self.__batch_size = cfg.DET.BATCH_SIZE

		"""
		Class objects
		"""

		# Only load weights if we're supposed to
		if load_weights:
			# Modified RetinaNet rotated bounding box method (from Jing)
			if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.ROTATED_RETINANET:
				self.__detector = RotatedRetinaNetWrapper()

			# Faster R-CNN via detectron
			if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.FASTER_RCNN_DETECTRON:
				# Define the detector
				self.__detector = FasterRCNNWrapper()

			# YOLOv3 via PyTorch implementation
			if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.PYTORCH_YOLOV3:
				# Define the detector
				self.__detector = YoloPyTorchWrapper()

			# Darknet python wrapper uses compiled libdarknet.so file
			if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.YOLOv3_DARKNETWRAPPER:
				# Where darknet can be found
				self.__darknet_loc = "/home/will/work/1-RA/src/Detector/darknet-alexeyAB/"

				# The compiled darknet .so library
				self.__DLL_loc = os.path.join(self.__darknet_loc, "libdarknet.so")

				# Network architecture definition
				self.__network_def = os.path.join(self.__darknet_loc, "cfg/yolov3-detector.cfg")

				# Where to find network weights
				self.__weights_loc = os.path.join(self.__darknet_loc, "backup/detector_augmented/yolov3-detector_best.weights")
				
				# Where to find information about the data
				self.__meta_loc = os.path.join(self.__darknet_loc, "cfg/detector.data")
				
				# Define the detector
				self.__detector = DarknetWrapper(	self.__DLL_loc, 
													self.__network_def, 
													self.__weights_loc, 
													self.__meta_loc		)

			# Python wrapper from - https://github.com/madhawav/YOLO3-4-Py
			if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.YOLO3_PY:
				# Define the detector
				self.__detector = Detector(	bytes(self.__network_def, encoding="utf-8"), 
											bytes(self.__weights_loc, encoding="utf-8"), 
											0,
	                   						bytes(self.__meta_loc, encoding="utf-8")		)

		"""
		Class setup
		"""

	"""
	Public methods
	"""

	def detectBatch(self, images):
		""" Given a list of images (a batch), detect on each one """

		# Detect on the given batch of images
		results = self.__detector.detect(images)

		return results

	# Given a single image, detect cattle in the image and return corresponding bounding boxes
	def detectSingle(self, img):
		# Act differently depending on the detection method
		if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.ROTATED_RETINANET:
			# Detect rotated bounding boxes
			results = self.__detector.detect()

			# Convert rboxes to common format?

			return results

		if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.FASTER_RCNN_DETECTRON:
			# Detect away
			results = self.__detector.detectSingle(img)

			# Convert instances to CPU from a tensor
			instances = results["instances"].to("cpu").get_fields()

			# Convert tensors to a list of numpy arrays
			boxes = [x.numpy().astype(int).tolist() for x in instances['pred_boxes']]
			scores = [x.numpy().astype(float).tolist() for x in instances['scores']]
			assert len(boxes) == len(scores)

			# The converted dictionary of detections
			converted = {'objects': []}

			# Iterate dually through the boxes and scores
			for box, score in zip(boxes, scores):
				x1 = int(box[0])
				y1 = int(box[1])
				x2 = int(box[2])
				y2 = int(box[3])
				converted['objects'].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': score})

			return converted

		if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.PYTORCH_YOLOV3:
			# Detect using the wrapper
			results = self.__detector.detect(img)

			print(results)
			input()

		if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.YOLOv3_DARKNETWRAPPER:
			# Detect away (using wrapper)
			results = self.__detector.detect(img)

			# Convert results into a common format
			detections = [x[2] for x in results]

			return detections

		if cfg.DET.DETECTION_METHOD == cfg.DETECTORS.YOLO3_PY:
			# Create an image instance in the correct format
			dark_frame = Image(img)

			# Detect
			results = self.__detector.detect(dark_frame)

			# Cleanup
			del dark_frame

			# Now convert into a dict that is consistent
			converted = {'objects': []}
			for cat, score, bounds in results:
				x1 = int(bounds[0] - bounds[2]/2)
				y1 = int(bounds[1] - bounds[3]/2)
				x2 = x1 + int(bounds[2])
				y2 = y1 + int(bounds[3])
				converted['objects'].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': score})

			return converted

	# Run the detector on a text file containing a list of testing images
	def testDetector(self, test_file, fold=-1, save_images=False, visualise=False, darknet_annos=False):
		# Make sure there's actually something there and it's of the correct form
		assert os.path.isfile(test_file) and test_file.endswith(".txt")

		# Create a list of testing files from this
		test_files = [line.rstrip() for line in open(test_file, "r")]

		# Directory to write testing detections out to
		test_det = "/home/will/work/1-RA/src/Detector/data/test_detections/"

		# Maintain a list of detections and ground truthes
		preds = []
		GTs = {}

		# Let's go through each test_file and detect on it
		for file in tqdm(test_files):
			# Load the image
			img = cv2.imread(file)

			# Detect and time it
			start = time.time()
			dets = self.detect(img)
			# print(f"Detector took {time.time() - start}s on: {file}")

			# Which annotation file style are we inspecting?
			if darknet_annos:
				# Find the corresponding GT via the annotation file
				anno_file = file.replace(".jpg", ".txt")
				assert os.path.isfile(anno_file) and anno_file.endswith(".txt")
				# Extract the annotations
				GT = DataUtils.readDarknetAnnotation(anno_file)
			# Use XML/VOC style
			else:
				# Find the corresponding GT via the annotation file
				anno_file = file.replace("/images/", "/labels-xml/")
				anno_file = anno_file.replace(".jpg", ".xml")

				assert os.path.isfile(anno_file) and anno_file.endswith(".xml")

				# Extract the annotations
				GT = DataUtils.readXMLAnnotation(anno_file)

			# There could be a problem with the annotation
			if GT['filename'] != os.path.basename(file):
				print(GT['filename'])
				print(os.path.basename(file))

			# Add things to the dictionary
			GTs[GT['filename']] = GT

			# Go through each detection that was returned
			for det in dets['objects']:
				# Add the image filename to each detection
				det['filename'] = os.path.basename(file)

				# Add it to the list
				preds.append(det)

			# Plot the detection if we're supposed to
			if visualise: det_img, _ = VM.plotDetections(img, dets, display=visualise, TP_only=False, GT_boxes=GT)

			# Save the detections to file if we're supposed to
			if save_images: cv2.imwrite(os.path.join(test_det, os.path.basename(file)), det_img)

		# Choose the save filename
		if fold >= 0: save_filename = f"fold-{fold}_to_be_evaluated.json"
		else: save_filename = "to_be_evaluated.json"

		# Pickle the list of GT and predictions so it can be evaluated
		with open(save_filename, 'w') as handle:
			json.dump({'GT': GTs, 'preds': preds}, handle, indent=1)

		# # Let's evaluate our performance
		mAP = EvaluationUtils.VOC_evaluate(GTs, preds)

		# Report this to the user
		print(f"mAP = {mAP}")

	# Run the detector live on a camera stream (e.g. webcam)
	def liveDetect(self, cam_num=0):
		# Get the camera stream
		cap = cv2.VideoCapture(cam_num)

		# Let's detect forever!
		while True:
			# Start timing
			start = time.time()

			# Capture frame-by-frame
			ret, frame = cap.read()

			# Detect on it
			dets = self.detect(frame)

			# Render detections
			det_img, _ = VM.plotDetections(frame, dets, display=False)

			# Display the resulting frame
			cv2.imshow("Detector", det_img)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

			# Report the FPS
			print(f"FPS = {1/(time.time() - start)}")

		# Cleanup
		cap.release()
		cv2.destroyAllWindows()

	# Run the detector on a video, save a new video to file with the rendered detections
	def detectOnVideo(self, video_dir):
		# Open the input video
		video = cv2.VideoCapture(video_dir)

		# Desired output dimensions
		out_w = 1280
		out_h = 720

		# Detection line thickness
		line_thickness = 2

		# Get video properties
		w, h, fps, length = ImageUtils.retrieveVideoProperties(video)

		print(f"Source video {w}x{h}@{fps}, total frames: {length}")

		# Open the output videowriter
		output_path = os.path.join(os.path.dirname(video_dir), f"{os.path.basename(video_dir)[:-4]}_output.avi")
		encoding = cv2.VideoWriter_fourcc('M','J','P','G')
		writer = cv2.VideoWriter(output_path, encoding, fps, (out_w, out_h))

		print(f"Writing out video to: {output_path}")

		# Read until the video is completed
		pbar = tqdm(total=length)
		while video.isOpened():
			# Get a frame
			_, frame = video.read()

			# Resize to the desired dimensions
			frame = cv2.resize(frame, (out_w, out_h))

			# Detect on the frame
			dets = self.detect(frame)

			# Render the detections
			det_img, _ = VM.plotDetections(frame, dets, display=False, thickness=line_thickness)

			# Write the rendered frame out to the video file
			writer.write(det_img)

			# Management stuff
			cv2.waitKey(1)
			pbar.update()
		pbar.close()
		writer.release()

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

# Entry method/unit testing method
if __name__ == '__main__':
	# Create an instance
	DM = DetectionManager()

	# Test a detector
	# img_fp = "/home/will/work/1-RA/src/Detector/data/consolidated_augmented/000001.jpg"
	# DM.detect(cv2.imread(img_fp))

	# Darknet testing file
	# test_file = "/home/will/work/1-RA/src/Detector/data/consolidated_augmented/0test.txt"
	# test_file = "/home/will/work/1-RA/src/Detector/data/CEADetection/0-valid.txt"
	fold = 9
	test_file = f"/work/ca0513/datasets/CEADetection/{fold}-test.txt"

	# Let's test the detector on the darknet test files
	DM.testDetector(test_file, fold=fold, save_images=False, visualise=False)

	# Test the detector live via a webcam
	# DM.liveDetect()

	# Test the detector on a video, save a new video to file with the rendered detections
	# video_dir = "/home/will/work/1-RA/src/Detector/data/videos/7.mov"
	# DM.detectOnVideo(video_dir)