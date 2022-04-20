#!/usr/bin/env python

# Core libraries
import os
import sys
# sys.path.append("/home/will/work/1-RA/src")
sys.path.append("../../")
import cv2
import argparse
import numpy as np
from tkinter import *
from tkinter.ttk import *

# My libraries
from src.IDButtonGridGUI import IDButtonGridGUI
from src.QueryImageGUI import QueryImageGUI
from src.ImagePreviewGUI import ImagePreviewGUI
from src.FileHandler import FileHandler

"""
Note: optimised ONLY for 1920x1080p display
To optimise for a new sized display, change variables:
	self.__max_cols = w
	self.__max_rows = h
below in __init__(self, args):
"""

class IdentificationGUI(object):
	# Class constructor
	def __init__(self, args):
		"""
		Class attributes
		"""

		# The name of our GUI window
		self.__window_name = "Will\'s incredible labelling program"

		# Maximum buttons per row, and max number of rows (optimised for a 1920x1080 disp)
		self.__max_cols = 12
		self.__max_rows = 8

		"""
		Class objects
		"""

		# Create our file handler object
		self.__file_handler = FileHandler(args.input_folder, args.output_folder)

		# Master GUI object
		self.__master = Tk()

		"""
		Class setup
		"""

		# Setup our GUI
		self.__initInterface()

	"""
	Public methods
	"""

	# Start the labelling interface
	def start(self):
		# Start the main tk loop
		self.__master.mainloop()

	"""
	(Effectively) private methods
	"""

	# Initialise the core interface
	def __initInterface(self):
		# Set the title
		self.__master.title(self.__window_name)

		# Make it non-resizable
		self.__master.resizable(0, 0)

		# Bind keyboard events
		self.__master.bind("<Left>", lambda event: self.__keyPressedCallback(True))
		self.__master.bind("<Right>", lambda event: self.__keyPressedCallback(False))
		self.__master.bind("a", lambda event: self.__keyPressedCallback(True))
		self.__master.bind("d", lambda event: self.__keyPressedCallback(False))

		# Create a frame for the query image and image preview
		self.__left_container = Frame(self.__master)
		self.__left_container.pack(side=LEFT)

		# Create a frame for the button interface
		self.__right_container = Frame(self.__master)
		self.__right_container.pack(side=RIGHT)

		# Create the query image subinterface
		self.__query_interface = QueryImageGUI(	self.__left_container,
												self.__file_handler.getQueryImage(),
												self.__file_handler.getTotalCount()		)

		# Create the image preview subinterface
		self.__preview_interface = ImagePreviewGUI(	self.__left_container	)

		# Create the tiled buttons subinterface
		self.__button_interface = IDButtonGridGUI(	self.__right_container,
													self.__file_handler.getClassExamples(),
													self.__IDButtonCallback,
													self.__prevNextImageButtonCallback,
													self.__mouseOverIDCallback,
													self.__max_cols,
													self.__max_rows				)

	"""
	Private callback functions
	"""

	# Called when a keyboard key is pressed (change pages)
	def __keyPressedCallback(self, left):
		self.__button_interface.prevNextPageButtonCallback(left)

	# Called when an identity button is pressed (including the add identity or "I don't know" buttons)
	def __IDButtonCallback(self, class_ID):
		# We pressed on an ID
		if class_ID in self.__file_handler.getDataset().keys():
			# Move the image across
			self.__file_handler.positiveID(class_ID)

			# We may need to enable the prev-next image buttons for that class
			self.__button_interface.enablePrevNext(class_ID)

			# Get an example image for ID
			example = self.__file_handler.getClassExamples()[class_ID]

		# The user pressed the new ID button
		elif class_ID == "new-ID":
			# Handle the file sides of things (create a new folder, move the image across)
			new_ID = self.__file_handler.createNewID()

			# Get an example image for this new ID
			new_example = self.__file_handler.getClassExamples()[new_ID]

			# Update the buttons and pages to reflect the new ID
			self.__button_interface.addNewIdentity(new_ID, new_example)

		# The user pressed the "I don't know button!"
		elif class_ID == "unsure":
			# Move the image across to the unsure category
			self.__file_handler.unsure()

		# See how many images we have left
		current_count, total_count = self.__file_handler.getProgress()

		# Report if we've finished to the console
		if current_count == total_count: print("Labelling finished!")

		# Update the query image
		self.__query_interface.update(	self.__file_handler.getQueryImage(),
										current_count,
										total_count 						)

	# Called when the next/prev image button is pressed for a particular ID
	def __prevNextImageButtonCallback(self, prev, ID):
		# The previous button was pressed
		if prev: 
			increment = -1
			print(f"Getting previous image for ID: {ID}")
		# The next button was pressed
		else:
			increment = 1
			print(f"Getting next image for ID: {ID}")

		# Get a new example image for this ID
		np_image = self.__file_handler.getNewExample(ID, increment)

		# Actually update the button image
		self.__button_interface.updateButtonImage(ID, np_image)

		# Update the image preview correspondingly
		self.__preview_interface.updateImage(np_image)

	# Called when the mouse is over the container for a particular ID
	def __mouseOverIDCallback(self, entering, ID):
		if entering:
			# Get the image for this ID
			np_image = self.__button_interface.getCurrentImageForID(ID)

			# Use it to update the preview
			self.__preview_interface.updateImage(np_image)
		else:
			self.__preview_interface.loseImage()

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
	# Default data source locations
	input_folder = "D:\\Work\\Data\\RGBDCows2020\\identification\\tobelabelled"
	output_folder = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\output"
	# input_folder = "data/input"
	# output_folder = "data/output"

	# Parse any arguments
	parser = argparse.ArgumentParser(description="Labelling parameters")
	parser.add_argument('--input_folder', type=str, default=input_folder,\
						help="Input folder containing all images to be labelled/identified")
	parser.add_argument('--output_folder', type=str, default=output_folder,\
						help="Output folder either empty or with images labelled to this point")
	args = parser.parse_args()

	# Create an instance and start the labelling GUI
	gui = IdentificationGUI(args)
	gui.start()