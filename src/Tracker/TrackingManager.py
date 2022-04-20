#!/usr/bin/env python

# Core libraries

# My libraries
from pyKCF.KCFTracker import KCFTracker

"""
Descriptor
"""

class TrackingManager(object):
	# Class constructor
	def __init__(self, ):
		"""
		Class attributes
		"""

		"""
		Class objects
		"""

		# The list of trackers
		self.__trackers = []

		"""
		Class setup
		"""

	"""
	Public methods
	"""

	def initialise(self, image, regions):
		# Iterate over the regions
		for region in regions:
			# Select the correct tracking method
			if cfg.TRACK.TRACKING_METHOD == cfg.TRACKERS.KCF:
				tracker = KCFTracker()
			elif cfg.TRACK.TRACKING_METHOD == cfg.TRACKERS.LKT:
				pass
			else:
				Utility.die("Tracking method not recognised", __file__)

			# Initialise it
			tracker.initialise(image, RoI)

			# Add it to the list
			self.__tracker.append(tracker)

	# Update all the trackers
	def update(self, image):
		return [x.update() for x in self.__trackers]

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
	