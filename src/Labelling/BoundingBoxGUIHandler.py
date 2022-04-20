#!/usr/bin/env python

import cv2
import numpy as np

class BoundingBoxGUIHandler:
	def __init__(self, window_name, window_width):
		self._acceptedBBoxes = []

		self._window_name = window_name

		self._window_width = window_width

		self._clicked = False

		self._P1 = (0, 0)

		self._P2 = (0, 0)

		self._bbox_thickness = 1

		self._scale_factor = 1

	def constructLabellingInterface(self, input_image):
		display_image = input_image.copy()

		height, width = display_image.shape[:2]

		self._scale_factor = self._window_width / float(width)
		new_height = int(self._scale_factor * height)

		return cv2.resize(display_image, (self._window_width, new_height))

	def scaleBoundingBoxesToOriginalSize(self):
		for x in range(len(self._acceptedBBoxes)):
			elem0 = int(self._acceptedBBoxes[x][0][0] / self._scale_factor)
			elem1 = int(self._acceptedBBoxes[x][0][1] / self._scale_factor)
			elem2 = int(self._acceptedBBoxes[x][1][0] / self._scale_factor)
			elem3 = int(self._acceptedBBoxes[x][1][1] / self._scale_factor)
			self._acceptedBBoxes[x]  = ((elem0, elem1), (elem2, elem3))

	def onMouseCallback(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self._clicked = True
			self._P1 = (x, y)
			self._P2 = (x, y)
		elif event == cv2.EVENT_LBUTTONUP:
			self._clicked = False
			self._P2 = (x, y)
		elif event == cv2.EVENT_MOUSEMOVE:
			if self._clicked:
				self._P2 = (x, y)

	def updateAndShowImage(self, display_image):
		draw_image = display_image.copy()

		cv2.rectangle(draw_image, self._P1, self._P2, (0, 255, 0), self._bbox_thickness)

		for box in self._acceptedBBoxes:
			cv2.rectangle(draw_image, box[0], box[1], (255, 0, 0), self._bbox_thickness)

		cv2.imshow(self._window_name, draw_image)


	def clearAllBoundingBoxes(self):
		self._acceptedBBoxes = []
		self._P1 = (0, 0)
		self._P2 = (0, 0)

	def labelImage(self, image, image_name):
		# Do some setting up
		self.clearAllBoundingBoxes()

		display_image = self.constructLabellingInterface(image)

		cv2.setMouseCallback(self._window_name, self.onMouseCallback)

		print '\nLet\'s label this image = {:s}'.format(image_name)
		print '(g): accept drawn bounding box (good)'
		print '(d): this image is fully labelled (done)'
		print '(m): mistake made, start over'
		print '(ctrl+c): exit'

		# Main labelling loop
		while True:
			# Get current keypress
			c = cv2.waitKey(1)

			# Current bounding box is deemed good by the user, accept it
			if c ==  ord('g'):
				if self._P1[0] == 0 and self._P1[1] == 0 and self._P2[0] == 0 and self._P2[1] == 0:
					print 'Draw a rectangle first before accepting it'
				else:
					self._acceptedBBoxes.append((self._P1, self._P2))

					print 'Labelled bounding box #{:d} has been accepted'.format(len(self._acceptedBBoxes))
			# User thinks they're done labelling this image, present a "are you sure" type thing
			elif c == ord('d'):
				if len(self._acceptedBBoxes) == 0:
					print 'You haven\'t accepted any bounding boxes, you\'re not quite done yet'
				else:
					print 'You\'ve said you\'re done, are you sure? (y/n)'

					while True:
						c = cv2.waitKey(1)

						if c == ord('y'):
							print 'Awesome, image={:s} finished'.format(image_name)

							self.scaleBoundingBoxesToOriginalSize()

							return self._acceptedBBoxes
						elif c == ord('n'):
							print 'Oops, let\'s label some more'
							break
			# User thinks they've made a mistake, start over (erase all bounding boxes)
			elif c == ord('m'):
				print 'Whoops, you\'ve made a mistake, starting again for this image'

				self.clearAllBoundingBoxes()

			self.updateAndShowImage(display_image)