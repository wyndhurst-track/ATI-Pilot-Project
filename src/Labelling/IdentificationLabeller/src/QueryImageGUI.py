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
Class implements the query image GUI
"""

class QueryImageGUI():
	# Class constructor
	def __init__(self, super_interface, query_image, total_count):
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
		self.__initInterface(query_image, total_count)

	"""
	Public methods
	"""

	# Update the GUI
	def update(self, np_image, current_count, total_count, initial=False):
		self.__updateImage(np_image, initial=initial)
		self.__updateProgress(current_count, total_count, initial=initial)

	"""
	(Effectively) private methods
	"""

	# Update the image we're drawing
	def __updateImage(self, np_image, initial=False):
		# If there's no image, we're likely finished
		if np_image is None:
			self.__query_image = self.__blank_img
		else:
			# Convert to BGR from RGB
			np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

			# Rescale proportionally to the size we're after
			np_image = ImageUtils.proportionallyResizeImageToMax(np_image, self.__max_img_w, self.__max_img_h)

			# Insert it at the top of the black image (so the label size stays consistent)
			padded_np_image = self.__np_blank_image.copy()
			padded_np_image[0:np_image.shape[0], 0:np_image.shape[1], :] = np_image

			# Convert it to TK format
			self.__query_image = ImageUtils.convertImage(padded_np_image)

		# Update the image if this isn't the initial run
		if not initial: self.__image_label.config(image=self.__query_image)

	# Update our progress
	def __updateProgress(self, current_count, total_count, initial=False):
		percentage = (current_count / float(total_count)) * 100
		self.__progress_label_text = f"{current_count}/{total_count}={percentage:.2f}% completed."
		
		# If this isn't the initial run, update the text
		if not initial: self.__progress_label.config(text=self.__progress_label_text)

	# Draw our interface
	def __initInterface(self, query_image, total_count):
		# Create our main frame
		self.__frame = Frame(self.__super_interface, relief=RAISED, borderwidth=2)
		self.__frame.pack(side=TOP, padx=5, pady=5)

		# Add a label to the top
		self.__label = Label(self.__frame, text="Query image")
		self.__label.pack(side=TOP, padx=5, pady=5)

		# Update the query image and progress label text
		self.update(query_image, 0, total_count, initial=True)

		# Add the image
		self.__image_label = Label(self.__frame, image=self.__query_image)
		self.__image_label.pack(side=TOP, padx=5, pady=5)

		# Add a label telling us our progress
		self.__progress_label = Label(self.__frame, text=self.__progress_label_text)
		self.__progress_label.pack(side=TOP, padx=5, pady=5)