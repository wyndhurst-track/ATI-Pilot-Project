#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import xml.etree.ElementTree as ET

# My libraries
from config import cfg
from Cameras.Camera import Camera

"""
Descriptor
"""

class CameraManager(object):
	# Class constructor
	def __init__(self, camera_ID):
		"""
		Class attributes
		"""

		# The camera ID this class instance refers to
		self.__camera_ID = camera_ID

		# The file at which constant data is stored about each camera
		self.__camera_data_fp = os.path.join(cfg.GEN.BASE_DIR, "data/camera_data.xml")

		"""
		Class objects
		"""

		# The dictionary of camera objects, the key is its ID
		self.__cameras = {}

		"""
		Class setup
		"""

		# Connect to RSTP stream?

		self.__populateCameras()

	"""
	Public methods
	"""

	# 
	def retrieveInput(self):
		pass

	def process(self):
		# Iterate through each camera and instruct them to process
		for ID in sorted(self.__cameras.keys()):
			self.__cameras[ID].process()

	# Plot on an image of the barn the positional distribution of cameras and the associated adjacencies
	def plotCameras(self):
		pass

	# Simply print each cameras serial number
	def printSerialNumbers(self):
		for ID in sorted(self.__cameras.keys()):
			print(f"For ID={ID}, serial={self.__cameras[ID].getSerialNumber()}")

	"""
	(Effectively) private methods
	"""

	# Populate the dictionary of camera objects with constant variables defined in file
	def __populateCameras(self):
		# Open the XML file
		tree = ET.parse(self.__camera_data_fp)

		# Iterate over each camera
		for camera_xml in tree.findall('camera'):
			self.__loadCameraData(camera_xml)

	def __loadCameraData(self, camera_xml):
		# Retrieve the ID
		ID = int(camera_xml.find('ID').text)

		# Retrieve the serial number
		serial_no = str(camera_xml.find('serial_no').text)

		# Get the camera type
		camera_type = str(camera_xml.find('type').text)

		# Get its GPS coordinates
		long_gps = float(camera_xml.find('longitudinal').text)
		lat_gps = float(camera_xml.find('latitudinal').text)

		# Get the list of camera IDs this camera is adjacent to
		# They're separated by commas, so delimit it and convert to integer
		adjacency = str(camera_xml.find('adjacency').text)
		adjacency = adjacency.split(',')
		adjacency = [int(x) for x in adjacency]

		# Create the camera object
		camera_obj = Camera(	ID,
								serial_no,
								camera_type,
								(long_gps, lat_gps),
								adjacency				)

		# Assign it to our dictionary
		self.__cameras[ID] = camera_obj

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
	cm = CameraManager()
	cm.printSerialNumbers()