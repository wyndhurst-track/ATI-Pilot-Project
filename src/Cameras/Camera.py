#!/usr/bin/env python

# Core libraries

# My libraries

"""
Descriptor
"""

class Camera(object):
	# Class constructor
	def __init__(	self, 
					ID, 
					serial_number,
					camera_type,
					gps_position,
					adjacency 		):
		"""
		Class attributes
		"""

		# The ID of this camera
		self.__ID = ID

		# The serial number of this camera
		self.__serial_no = serial_number

		# The camera type (e.g. normal, fisheye)
		self.__type = camera_type

		"""
		Class objects
		"""

		"""
		Class setup
		"""

	"""
	Public methods
	"""

	def process(self):
		pass

	"""
	(Effectively) private methods
	"""

	"""
	Getters
	"""

	# Return a list of positionally adjacent cameras in the barn
	def getAdjacency(self):
		pass

	def getID(self):
		return self.__ID
	def getSerialNumber(self):
		return self.__serial_no

	"""
	Setters
	"""

	"""
	Static methods
	"""


# Entry method/unit testing method
if __name__ == '__main__':
	pass