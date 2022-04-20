# Core libraries
import os
import sys
sys.path.append("../../../")
import numpy as np
from tkinter import *
from tkinter.ttk import *
from collections import OrderedDict

# My libraries
from Utilities.ImageUtils import ImageUtils

"""
Class ...
"""
class IDButtonGridGUI():
	# Class constructor
	def __init__(	self, 
					super_interface,
					button_images,
					IDButtonCallback,
					prevNextImageButtonCallback,
					mouseOverIDCallback,
					max_cols,
					max_rows 	):
		# Maintain a reference to the superior interface
		self.__super_interface = super_interface

		# Reference to ID button callback functions in superclass
		self.__ID_callback = IDButtonCallback
		self.__prev_next_callback = prevNextImageButtonCallback
		self.__mouse_over_callback = mouseOverIDCallback

		# Maximum buttons per row, and max number of rows
		self.__max_cols = max_cols
		self.__max_rows = max_rows

		# Maximum image size for buttons (in any dimension)
		self.__max_button_image_size = 100

		# Blank image that buttons are pasted into
		self.__np_template_img = np.zeros((	self.__max_button_image_size, 
											self.__max_button_image_size, 3), np.uint8)
		self.__np_template_img.fill(217)

		# Counter for page IDs
		self.__page_ID_ctr = 0

		# A list of pages objects (class at bottom of this file)
		self.__pages = []

		# The notebook object which handles all the tabs
		self.__notebook = Notebook(self.__super_interface)
		self.__notebook.pack()

		# Current page being viewed
		self.__current_page = 0

		# Get it ready
		self.__initInterface(button_images)

	"""
	Public methods
	"""

	def updateButtonImage(self, ID_name, np_image):
		# Resize the image as needed
		resized_img = ImageUtils.proportionallyResizeImageToMax(	np_image, 
																	self.__max_button_image_size, 
																	self.__max_button_image_size	)
		
		# Paste it into a copy of the template image
		button_img = self.__np_template_img.copy()
		button_img[0:resized_img.shape[0], 0:resized_img.shape[1], :] = resized_img

		# Convert it to PIL form
		conv_img = ImageUtils.convertImage(button_img)

		# Update the assets for this ID
		self.__pages[self.__current_page].setOriginalImage(ID_name, np_image)
		self.__pages[self.__current_page].setConvertedImage(ID_name, conv_img)

		# And update the actual button image
		self.__pages[self.__current_page].updateButtonImage(ID_name, conv_img)

	# The previous/next image buttons for a class may be disabled, if so, enable them
	def enablePrevNext(self, ID_name):
		# Update the assets for this ID
		self.__pages[self.__current_page].enablePrevNext(ID_name)

	# Called when the create new ID button is pressed to add a new ID
	def addNewIdentity(self, new_ID, new_example):
		new_page_needed = self.__pages[-1].addIDButton(new_ID, new_example)

		# The method may have returned that a new page is needed
		if new_page_needed: 
			self.__createNewPage()
			self.__notebook.add(self.__pages[-1], text=f"Page {self.getNumPages()}")
			self.__updatePageLabelText()

			# The prev/next buttons might need to be updated
			self.__next_page_button.config(state="normal")
			self.__prev_page_button.config(state="normal")

	"""
	(Effectively) private methods
	"""

	def __initInterface(self, button_images):
		# Create our main frame
		self.__frame = Frame(self.__super_interface, relief=RAISED, borderwidth=2)
		self.__frame.pack(padx=5, pady=5)

		self.__top_label = Label(self.__frame, text="Click the button corresponding to the query image")
		self.__top_label.pack(side=TOP, padx=5, pady=5)

		# Create the first page object
		self.__createNewPage()

		# Create all the ID buttons, may create new pages
		for class_ID, example in sorted(button_images.items()):
			# Add this button to the latest page
			new_page_needed = self.__pages[-1].addIDButton(class_ID, example)

			# The method may have returned that a new page is needed
			if new_page_needed: self.__createNewPage()

		# Create the special button for creating a new individual/ID
		text = "Create new category"
		self.__new_ID_button = Button(self.__frame, text=text, command=lambda x="new-ID": self.__ID_callback(x))
		self.__new_ID_button.pack(side=LEFT, padx=5, pady=5)

		# Create the button for the "I don't know" category
		text = "I don't know!"
		self.__dont_know_button = Button(self.__frame, text=text, command=lambda x="unsure": self.__ID_callback(x))
		self.__dont_know_button.pack(side=LEFT, padx=5, pady=5)

		# Setup the next and previous page buttons
		self.__next_page_button = Button(self.__frame, text="Next page >>", command=lambda: self.prevNextPageButtonCallback(False))
		self.__next_page_button.pack(side=RIGHT, padx=5, pady=5)
		self.__prev_page_button = Button(self.__frame, text="<< Prev page", command=lambda: self.prevNextPageButtonCallback(True))
		self.__prev_page_button.pack(side=RIGHT, padx=5, pady=5)
		
		# Grey out the buttons if there are < 1 pages
		if len(self.__pages) <= 1:
			self.__next_page_button.config(state=DISABLED)
			self.__prev_page_button.config(state=DISABLED)

		# Create the page label
		self.__page_label = Label(self.__frame, text="temp")
		self.__page_label.pack(side=RIGHT, padx=100)
		self.__updatePageLabelText()

		# Add all the pages to the notebook
		for i, page in enumerate(self.__pages):
			self.__notebook.add(page, text=f"Page {i+1}")

		# Enable movement between the pages
		self.__notebook.enable_traversal()

	# Create a new page
	def __createNewPage(self):
		# Create the new page
		page = PageFrame(	self.__page_ID_ctr,
							self.__max_button_image_size, 
							self.__mouse_over_callback,
							self.__pageVisibilityCallback,
							self.__prev_next_callback,
							self.__ID_callback,
							self.__max_cols,
							self.__max_rows,
							self.__np_template_img				)

		# Add it to the list of pages
		self.__pages.append(page)

		# Increment our ID counter
		self.__page_ID_ctr += 1

	def __updatePageLabelText(self):
		text_str = f"Viewing page {self.__current_page+1} of {len(self.__pages)}"

		# The label may not have been created yet
		try: self.__page_label.config(text=text_str)
		except Exception as e: pass
	"""
	Callbacks
	"""

	# Called when the next/previous page button is pressed
	def prevNextPageButtonCallback(self, previous):
		# Previous or next?
		if previous:
			# Wrap the indices around
			if self.__current_page == 0: self.__current_page = len(self.__pages) - 1
			else: self.__current_page -= 1 
		else:
			# Wrap the indices around
			if self.__current_page == len(self.__pages) - 1: self.__current_page = 0
			else: self.__current_page += 1 

		# Update the label text
		self.__updatePageLabelText()

		# Show the correct page
		self.__notebook.select(self.__pages[self.__current_page])

	# Called when a certain page becomes visibile (it is selected)
	def __pageVisibilityCallback(self, page_ID):
		# Update the current page pointer
		self.__current_page = page_ID

		# Update the page label
		self.__updatePageLabelText()

	"""
	Getters
	"""

	# Get the converted image currently being displayed for a particular ID
	def getCurrentImageForID(self, ID):
		return self.__pages[self.__current_page].getCurrentImageForID(ID)

	# Simply get the number of pages currently
	def getNumPages(self):
		return len(self.__pages)

"""
Additional class representing a page of ID buttons
"""
class PageFrame(ttk.Frame):
	# Class constructor
	def __init__(	self, 
					ID, 
					max_image_size, 
					mouseOverIDCallback, 
					on_visibility_callback,
					prev_next_callback,
					ID_callback,
					max_cols,
					max_rows,
					template_img			):
		# Init the superclass
		ttk.Frame.__init__(self)

		# The ID of this page
		self.__ID = ID

		# If this frame becomes visible, send our ID to a callback in the superclass
		self.bind("<Visibility>", lambda event, x=self.__ID: on_visibility_callback(x))

		# Reference to callback functions
		self.__mouse_over_callback = mouseOverIDCallback
		self.__prev_next_callback = prev_next_callback
		self.__ID_callback = ID_callback

		# Template image for buttons (just a gray square image)
		self.__np_template_img = template_img

		# Maximum image size for buttons (in any dimension)
		self.__max_button_image_size = max_image_size

		# Maximum buttons per row, and max number of rows (optimised for a 1920x1080 disp)
		self.__max_cols = max_cols
		self.__max_rows = max_rows

		# The current column and row
		self.__current_col = 0
		self.__current_row = 0

		# Dictionary of references to assets for each ID
		self.__asset_dict = {}

	# Called when a button is added to this page
	def addIDButton(self, ID_name, example):
		# Check whether we should be on a new row
		if self.__current_col >= self.__max_cols:
			self.__current_row += 1
			self.__current_col = 0

		# Create a dictionary entry storing a reference to everything we've done here
		assets = {}

		# Get the example image
		np_image = example['img']

		# Store the original image
		assets['orig_image'] = np_image

		# Proportionally rescale the button image
		np_image = ImageUtils.proportionallyResizeImageToMax(np_image, self.__max_button_image_size, self.__max_button_image_size)

		# Paste it into a copy of the template image
		button_img = self.__np_template_img.copy()
		button_img[0:np_image.shape[0], 0:np_image.shape[1], :] = np_image

		# Convert the image to the tk format via PIL
		assets['image'] = ImageUtils.convertImage(button_img)

		# Create a container frame for this ID
		assets['container'] = Frame(self)
		assets['container'].grid(row=self.__current_row, column=self.__current_col, sticky=N+S+E+W)
		assets['container'].bind("<Enter>", lambda event, x=ID_name: self.__mouse_over_callback(entering=True, ID=x))
		assets['container'].bind("<Leave>", lambda event, x=ID_name: self.__mouse_over_callback(entering=False, ID=x))

		# Create the ID button and add it to the container
		assets['ID_button'] = Button(	assets['container'], 
										text=f"ID = {ID_name}", 
										image=assets['image'], 
										command=lambda x=ID_name: self.__ID_callback(x), 
										compound=BOTTOM	)
		assets['ID_button'].pack(side=TOP)

		# Create the prev and next image buttons
		assets['next_img_button'] = Button(	assets['container'],
											text=">",
											command=lambda x=ID_name: self.__prev_next_callback(False, x))
		assets['prev_img_button'] = Button(	assets['container'], 
											text="<",
											command=lambda x=ID_name: self.__prev_next_callback(True, x))

		# Disable the prev/next buttons if this class has a single image currently associated with it
		if example['total'] == 1:
			assets['next_img_button'].config(state=DISABLED)
			assets['prev_img_button'].config(state=DISABLED)

		# Actually distribute the buttons
		assets['next_img_button'].pack(side=RIGHT)
		assets['prev_img_button'].pack(side=RIGHT)

		# Increment the column counter
		self.__current_col += 1

		# Add this to the dictionary of assets for this page
		self.__asset_dict[ID_name] = assets

		# Check whether we should indicate that a new page is needed
		if self.__current_row == self.__max_rows - 1 and self.__current_col == self.__max_cols:
			return True

		return False

	"""
	Getters
	"""

	def getCurrentImageForID(self, ID):
		return self.__asset_dict[ID]['orig_image']

	"""
	Setters
	"""

	def setOriginalImage(self, ID, np_image):
		self.__asset_dict[ID]['orig_image'] = np_image

	def setConvertedImage(self, ID, conv_img):
		self.__asset_dict[ID]['image'] = conv_img

	def updateButtonImage(self, ID, conv_img):
		self.__asset_dict[ID]['ID_button'].config(image=conv_img)

	def enablePrevNext(self, ID):
		self.__asset_dict[ID]['next_img_button'].config(state=NORMAL)
		self.__asset_dict[ID]['prev_img_button'].config(state=NORMAL)