#!/usr/bin/env python

# Core libraries
import cv2
import sys
import random

"""
Utility class for static methods
"""

class Utility:
	# Compute the IoU of two given rectangles
	@staticmethod
	def bboxIntersectionOverUnion(boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])

		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)

		# return the intersection over union value
		return iou

	# Die gracefully whilst printing the error method
	@staticmethod
	def die(message, file):
		print(f"\nERROR MESSAGE:_________________\n\"{message}\"\nin file: {file}\nExiting..")
		sys.exit(0)

# Entry method/unit testing
if __name__ == '__main__':
	pass