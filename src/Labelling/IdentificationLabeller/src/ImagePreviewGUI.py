# Core libraries
import os
import sys
sys.path.append("../../../")
import cv2
import numpy as np
from tkinter import *
from tkinter.ttk import *

# My libraries
from Utilities.ImageUtils import ImageUtils

"""
Class implements the preview image GUI
"""

class ImagePreviewGUI():
	# Class constructor
	def __init__(self, super_interface):
		# Maximum image size
		self.__max_img_w = 400
		self.__max_img_h = 400

		# Maintain a reference to the superior interface
		self.__super_interface = super_interface

		# Create the empty image (matches the interface background)
		self.__np_blank_image = np.zeros((self.__max_img_w, self.__max_img_h, 3), np.uint8)
		self.__np_blank_image.fill(217)
		self.__blank_img = ImageUtils.convertImage(self.__np_blank_image)

		# Get this interface ready
		self.__initInterface()

	"""
	Public methods
	"""

	# Called when the mouse enters an ID, display the image larger
	def updateImage(self, np_image):
		# Resize the image
		np_image = ImageUtils.proportionallyResizeImageToMax(np_image, self.__max_img_w, self.__max_img_h)

		# Insert it at the top of the black image (so the label size stays consistent)
		padded_np_image = self.__np_blank_image.copy()
		padded_np_image[0:np_image.shape[0], 0:np_image.shape[1], :] = np_image

		# Update the stored image
		self.__image = ImageUtils.convertImage(padded_np_image)

		# Update the label
		self.__image_label.config(image=self.__image)

	# Called when the mouse leaves a particular ID, just display a black empty image
	def loseImage(self):
		self.__image_label.config(image=self.__blank_img)

	"""
	(Effectively) private methods
	"""

	def __initInterface(self):
		# Create our main frame
		self.__frame = Frame(self.__super_interface, relief=RAISED, borderwidth=2)
		self.__frame.pack(side=TOP, padx=5, pady=5)

		# Add a label to the top
		self.__label = Label(self.__frame, text="Image Preview")
		self.__label.pack(side=TOP, padx=5, pady=5)

		# Create the image label
		self.__image_label = Label(self.__frame, image=self.__blank_img)
		self.__image_label.pack(side=TOP, padx=5, pady=5)