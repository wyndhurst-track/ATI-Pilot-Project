# Core libraries
import os
import sys
sys.path.append("../")

# My libraries
from config import cfg
from Utilities.DataUtils import DataUtils
from Utilities.ImageUtils import ImageUtils
from Visualiser.VisualisationManager import VisualisationManager

class ObservationManager(object):
	"""
	Class represents a group of observations for a particular video file
	"""

	def __init__(self, process_ID, camera_ID, filename, detector, identifier, database):
		""" Class constructor """

		# The process ID that is processing this
		self.__process_ID = process_ID

		# The camera ID that generated this series of observations
		self.__camera_ID = camera_ID

		# The filename for this video file
		self.__filename = filename

		# Reference to the detector network
		self.__detector = detector

		# Reference to the identifier network
		self.__identifier = identifier

		# Reference to the database manager
		self.__database = database

		# The list of UN-processed observations
		self.__unprocessed_queue = []

	"""
	Public methods
	"""

	def addUnprocessedObservation(self, frame, frame_ID, timestamp):
		""" Add an observation object that requires processing """

		# Create an obsevation object and add it to the process queue
		self.__unprocessed_queue.append(Observation(frame, frame_ID, timestamp))

	def processQueue(self):
		""" Process the queue of unprocessed observations """

		# Detect objects within the list of observations
		self.__detect()

		# Identify those detected objects
		self.__identify()

		# Write the observations to the database
		self.__write()

		# Clear the processing queue
		self.__unprocessed_queue = []

	"""
	(Effectively) private methods
	"""

	def __detect(self, visualise=0):
		""" For every observation in the queue, detect cow objects """

		# Collate the observations into a list of images
		images = [obvs.getFrame() for obvs in self.__unprocessed_queue]

		# Predict on this batch of images
		output = self.__detector.detectBatch(images)

		# Iterate through the set of detections for each observation
		for i, detections in enumerate(output):
			# Indicate the observation to add those predictions
			self.__unprocessed_queue[i].processDetections(detections)

			# Should we visualise each detection?
			if visualise:
				VisualisationManager.drawRotatedBbox(images[i], detections, display=True)

	def __identify(self):
		""" Identify each object within the unprocessed queue of observations """

		# Collate the RoIs into a 1D list
		RoIs = []
		for observation in self.__unprocessed_queue:
			for obj in observation.getObjects():
				RoIs.append(obj.getRoI())

		# Split this into a chunks of batch size
		batched = DataUtils.chunks(RoIs, cfg.ID.BATCH_SIZE)

		# Get the outputs for each batch
		outputs = [self.__identifier.identifyBatch(batch) for batch in batched]

		# Convert the outputs to a 1D list
		outputs = [item for sublist in outputs for item in sublist]

		# Iterate through the queue and objects and reassign the identities in the same order
		i = 0
		for observation in self.__unprocessed_queue:
			for obj in observation.getObjects():
				obj.setID(outputs[i])
				i += 1

	def __write(self):
		""" Write the complete observations to file via the database """

		# Iterate through each observation being processed
		for observation in self.__unprocessed_queue:
			# Iterate through each detected object
			for obj in observation.getObjects():
				# Record an instance
				print("record!")
				self.__database.record(		self.__camera_ID,
											obj, 
											observation.getTimestamp(),
											observation.getFrameID(),
											self.__filename				)

	"""
	Getters
	"""

	def getProcessQueueLength(self):
		""" Return the number of unprocessed observations """
		return len(self.__unprocessed_queue)

class Observation(object):
	"""
	Class represents a single image frame that may contain an object (cow)
	"""

	def __init__(self, frame, frame_ID, timestamp):
		"""
		Class constructor
		"""

		# The image for this observation
		self.__frame = frame

		# The frame ID for this frame into the video
		self.__frame_ID = frame_ID

		# The datetime object for this observation
		self.__timestamp = timestamp

		# The list of objects in this observation
		self.__objects = []

	"""
	Public methods
	"""

	def processDetections(self, detections):
		""" Given a set of detections for this observation, add the respective objects """

		# Iterate through each detection object
		for det in detections:
			# Extract the object
			cx = det['cx']
			cy = det['cy']
			w = det['w']
			h = det['h']
			angle = det['angle']
			score = det['score']

			# Extract the RoI based on this
			RoI = ImageUtils.extractRotatedSubImage(self.__frame, det)

			# Add the object to the observation
			self.__addObject(cx, cy, w, h, angle, score, RoI)

	def isEmpty(self):
		""" Is this observation devoid of an object? """
		if len(self.__objects) == 0: return True
		else: return False 

	"""
	(Effectively) private methods
	"""

	def __addObject(self, cx, cy, w, h, angle, score, RoI):
		""" Add a detected object to this observation """

		# Create the object
		obj = Object(cx, cy, w, h, angle, score, RoI)

		# Add it to the list for this observation
		self.__objects.append(obj)

	"""
	Getters
	"""

	def getObjects(self):
		return self.__objects

	def getFrame(self):
		return self.__frame

	def getFrameID(self):
		return self.__frame_ID

	def getTimestamp(self):
		return self.__timestamp

	"""
	Setters
	"""

	"""
	Static methods
	"""

class Object(object):
	""" Class represents a single object present within the observation """

	def __init__(self, cx, cy, w, h, angle, score, RoI):
		""" Class constructor """
		# Centrepoint of the object (in pixels)
		self.__cx = cx
		self.__cy = cy
		# The width and height of the object (in pixels)
		self.__w = w
		self.__h = h
		# The orientation of the object
		self.__angle = angle
		# The confidence score in this detected object
		self.__score = score
		# Extract the image RoI for this object
		self.__RoI = RoI

	"""
	Getters
	"""

	def getRoI(self):
		return self.__RoI
	def getBox(self):
		return self.__cx, self.__cy, self.__w, self.__h, self.__angle
	def getScore(self):
		return self.__score
	def getID(self):
		return self.__ID

	"""
	Setters
	"""

	def setID(self, ID):
		self.__ID = ID

# Entry method/unit testing method
if __name__ == '__main__':
	pass
