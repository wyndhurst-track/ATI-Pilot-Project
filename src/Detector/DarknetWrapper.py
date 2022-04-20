#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("/home/will/work/1-RA/src")
from ctypes import *
import cv2
import time

# My libraries
from Visualiser.VisualisationManager import VisualisationManager as VM

"""
This class wraps up darknet in python, provided there as a DLL of it
"""

# Darknet classes
class BOX(Structure):
	_fields_ = [("x", c_float),
				("y", c_float),
				("w", c_float),
				("h", c_float)]

class DETECTION(Structure):
	_fields_ = [("bbox", BOX),
				("classes", c_int),
				("prob", POINTER(c_float)),
				("mask", POINTER(c_float)),
				("objectness", c_float),
				("sort_class", c_int)]

class IMAGE(Structure):
	_fields_ = [("w", c_int),
				("h", c_int),
				("c", c_int),
				("data", POINTER(c_float))]

class METADATA(Structure):
	_fields_ = [("classes", c_int),
				("names", POINTER(c_char_p))]

class DarknetWrapper(object):
	# Class constructor
	def __init__(self, DLL, network, weights, meta):
		"""
		Class attributes
		"""

		self.__DLL_loc = DLL
		self.__network_def = bytes(network, encoding="utf-8")
		self.__weights_loc = bytes(weights, encoding="utf-8")
		self.__meta_loc = bytes(meta, encoding="utf-8")

		"""
		Class objects
		"""

		"""
		Class setup
		"""

		self.__setup()

	"""
	Public methods
	"""

	# Call YOLOv3 to detect on the given image
	def detect(self, image_array, nms=.45):
		# Convert numpy array to an image YOLO expects
		image = DarknetWrapper.array_to_image(image_array)
		self.__rgbgr_image(image)

		# Get the prediction from the library
		self.__predict_image(self.__net, image)

		num = c_int(0)
		pnum = pointer(num)
		dets = self.__get_network_boxes(self.__net, image.w, image.h, 0.5, 0.5, None, 0, pnum, 0)
		num = pnum[0]
		if (nms): self.__do_nms_sort(dets, num, self.__meta.classes, nms)

		res = []
		for j in range(num):
			for i in range(self.__meta.classes):
				if dets[j].prob[i] > 0:
					b = dets[j].bbox
					res.append((self.__meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
		res = sorted(res, key=lambda x: -x[1])

		# Free memory
		#self.__free_image(image)
		self.__free_detections(dets, num)

		return res

	"""
	(Effectively) private methods
	"""

	def __setup(self):
		self.__setupFunctions()
		self.__net = self.__load_net(self.__network_def, self.__weights_loc, 0)
		self.__meta = self.__load_meta(self.__meta_loc)

	def __setupFunctions(self):
		lib = CDLL(self.__DLL_loc, RTLD_GLOBAL)
		lib.network_width.argtypes = [c_void_p]
		lib.network_width.restype = c_int
		lib.network_height.argtypes = [c_void_p]
		lib.network_height.restype = c_int

		predict = lib.network_predict
		predict.argtypes = [c_void_p, POINTER(c_float)]
		predict.restype = POINTER(c_float)

		set_gpu = lib.cuda_set_device
		set_gpu.argtypes = [c_int]

		make_image = lib.make_image
		make_image.argtypes = [c_int, c_int, c_int]
		make_image.restype = IMAGE

		self.__get_network_boxes = lib.get_network_boxes
		self.__get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
		self.__get_network_boxes.restype = POINTER(DETECTION)

		make_network_boxes = lib.make_network_boxes
		make_network_boxes.argtypes = [c_void_p]
		make_network_boxes.restype = POINTER(DETECTION)

		self.__free_detections = lib.free_detections
		self.__free_detections.argtypes = [POINTER(DETECTION), c_int]

		free_ptrs = lib.free_ptrs
		free_ptrs.argtypes = [POINTER(c_void_p), c_int]

		network_predict = lib.network_predict
		network_predict.argtypes = [c_void_p, POINTER(c_float)]

		reset_rnn = lib.reset_rnn
		reset_rnn.argtypes = [c_void_p]

		self.__load_net = lib.load_network
		self.__load_net.argtypes = [c_char_p, c_char_p, c_int]
		self.__load_net.restype = c_void_p

		self.__do_nms_obj = lib.do_nms_obj
		self.__do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

		self.__do_nms_sort = lib.do_nms_sort
		self.__do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

		self.__free_image = lib.free_image
		self.__free_image.argtypes = [IMAGE]

		letterbox_image = lib.letterbox_image
		letterbox_image.argtypes = [IMAGE, c_int, c_int]
		letterbox_image.restype = IMAGE

		self.__load_meta = lib.get_metadata
		lib.get_metadata.argtypes = [c_char_p]
		lib.get_metadata.restype = METADATA

		self.__load_image = lib.load_image_color
		self.__load_image.argtypes = [c_char_p, c_int, c_int]
		self.__load_image.restype = IMAGE

		self.__rgbgr_image = lib.rgbgr_image
		self.__rgbgr_image.argtypes = [IMAGE]

		self.__predict_image = lib.network_predict_image
		self.__predict_image.argtypes = [c_void_p, IMAGE]
		self.__predict_image.restype = POINTER(c_float)

	"""
	Getters
	"""

	"""
	Setters
	"""

	"""
	Static methods
	"""

	# Convert a numpy image array to an image YOLO expects
	@staticmethod
	def array_to_image(arr):
		arr = arr.transpose(2,0,1)
		c = arr.shape[0]
		h = arr.shape[1]
		w = arr.shape[2]
		arr = (arr/255.0).flatten()
		data = DarknetWrapper.c_array(c_float, arr)
		im = IMAGE(w,h,c,data)
		return im

	@staticmethod
	def c_array(ctype, values):
		arr = (ctype*len(values))()
		arr[:] = values
		return arr

	@staticmethod
	def extractDarknetAnnotations(file):
		# Create a list of object annotations
		lines = [line.rstrip() for line in open(file, "r")]

		# List of annotations
		annotations = []

		# Iterate through the lines 
		for obj in lines:
			# Split into components
			split = obj.split(" ")

			# Extract the annotatino
			annotations.append([float(split[1]), float(split[2]), float(split[3]), float(split[4])])

		return annotations

# Entry method/unit testing method
if __name__ == '__main__':
	# Let's test the detector
	wrapper = DarknetWrapper()

	# Folder with a bunch of images to detect on
	image_folder = "/home/will/work/1-RA/detector/data/raw/BBC/1/images/"

	# Construct a list of image paths
	images = [os.path.join(image_folder, x) for x in sorted(os.listdir(image_folder)) if x.endswith(".jpg")]

	# Loop over them all
	for img_path in images:
		print(f"Detecting on {img_path}")
		img = cv2.imread(img_path)
		start = time.time()
		dets = wrapper.detect(img)
		print(f"Detector took {time.time() - start}s")
		VM.plotDetections(img, dets)
