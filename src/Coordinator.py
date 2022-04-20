# Core libraries
import os
import cv2
import sys
import time
import argparse
from datetime import timedelta

# My libraries
from config import cfg
from Detector.DetectionManager import DetectionManager
from Identifier.IdentificationManager import IdentificationManager
from Database.DatabaseManager import DatabaseManager
from Utilities.ImageUtils import ImageUtils
from Utilities.LoggingManager import LoggingManager
from Utilities.Observation import ObservationManager
from Utilities.FileSystemManager import FileSystemManager

class Coordinator(object):
	"""
	This class coordinates and oversees everything for the Wyndhurst production system. 
	"""

	def __init__(self, process_ID):
		""" Class constructor """

		# For a video being processed, how long to wait (in seconds) inbetween individual frames that need processing
		self.__frame_skip_time = cfg.COR.FRAME_SKIP_TIME

		# How long in seconds to wait for before checking again for an unprocessed file
		self.__check_wait_time = cfg.COR.CHECK_WAIT_TIME

		# Unique identifier for this process
		self.__process_ID = process_ID

		# Manages logging events (normal or failures) to file or the console
		self.__logging_manager = LoggingManager(self.__process_ID, to_console=cfg.GEN.LOG_TO_CONSOLE)

		# Manages interactions with files that need processing
		self.__file_system_manager = FileSystemManager(self.__process_ID, self.__logging_manager)

		# Manages the detection & localisation of cattle in imagery
		self.__detection_manager = DetectionManager()

		# Manages identifying cattle from detected regions
		self.__identification_manager = IdentificationManager()

		# Manages interactions with the database, writing findings
		self.__database_manager = DatabaseManager(self.__process_ID, self.__logging_manager, to_csv=cfg.DAT.TO_CSV)

	"""
	Public methods
	"""

	def initialise(self):
		""" Initialise things as necessary """

		# Indicate that we're setting up
		self.__logging_manager.logInfo(__file__, f"Beginning initialisation")
		
		# Ask the file system manager to ensure there aren't any files in the "processing" folder that may
		# not have been fully processed yet, if there are, move them back to be processed
		self.__file_system_manager.checkProcessing()

		# Indicate that we've finished setting up
		self.__logging_manager.logInfo(__file__, f"Finished initialising")

	def run(self):
		""" Main processing loop """

		# Indicate that we're about to start processing loop
		self.__logging_manager.logInfo(__file__, f"Beginning main processing loop")

		# Loop until we've been indicated to shutdown
		while not self.__keepRunning():
			# Ask the file manager to see if there is a new video file to be processed
			ret, filepath, camera_ID, timestamp = self.__file_system_manager.checkForUnprocessed()

			# It returned a filepath
			if ret:
				# Process the file at the given path
				self.__processFile(filepath, camera_ID, timestamp)

				# Processing has finished, indicate this to the file system manager
				self.__file_system_manager.finishedProcessing()

			# There wasn't anything
			else: 
				# Wait a bit and check again
				time.sleep(self.__check_wait_time)

				# Report that we're waiting for a file to process
				#self.__logging_manager.logInfo(__file__, f"Waited {self.__check_wait_time}s for new file.") ##### i did it

		# If we're here, we should shutdown components cleanly
		self.__cleanShutdown()

		# Indicate that we've finished successfully
		self.__logging_manager.logInfo(__file__, f"shutdown cleanly, exiting.")
		sys.exit(0)

	"""
	(Effectively) private methods
	"""

	def __processFile(self, filepath, camera_ID, timestamp):
		""" We have a file to be processed """

		# Extract the filename
		filename = os.path.basename(filepath)

		# Log that we have a file to process
		self.__logging_manager.logInfo(__file__, f"Beginning processing on: {filename}")

		# Create the group of observations for this file
		observations = ObservationManager(	self.__process_ID,
											camera_ID,
											filename,
											self.__detection_manager, 
											self.__identification_manager, 
											self.__database_manager			)

		# Open the input video
		video = cv2.VideoCapture(filepath)

		# Get video properties
		w, h, fps, length = ImageUtils.retrieveVideoProperties(video)

		# Precompute the number of frames we need to skip based off this
		num_skip_frames = fps * self.__frame_skip_time

		# Keep a count of the frame ID
		frame_ctr = 0

		# Iterate through the video
		while video.isOpened():
			# Get a frame
			ret, frame = video.read()

			# Increment the frame counter
			frame_ctr += 1

			# Ensure it isn't an empty frame
			if ret:
				# Add this frame as an unprocessed observation
				observations.addUnprocessedObservation(frame, frame_ctr, timestamp)

				# If there are a sufficient number of images for a single processing batch, do it!
				if observations.getProcessQueueLength() == cfg.DET.BATCH_SIZE: #######dealwith max 16 iamges.
					observations.processQueue()

			# We've reached the end of the video, stop
			else: 
				break

			# Skip the pre-determined number of frames
			for i in range(num_skip_frames):
				ret, _ = video.read()

				# Increment the frame counter
				frame_ctr += 1

				# We've reached the end of the video, stop
				if not ret: break

			# Update the datetime object to reflect this change
			timestamp += timedelta(seconds=self.__frame_skip_time)

		# There may be some observations left that need processing
		observations.processQueue()

		# Log that we have a file to process
		self.__logging_manager.logInfo(__file__, f"Finished processing on: {filename}")

	def __keepRunning(self):
		""" Should the main processing loop keep running """

		# TBC

		return False

	def __cleanShutdown(self):
		""" Indicate objects to shutdown cleanly as necessary """

		self.__database_manager.close()

# Entry method/unit testing method
if __name__ == '__main__':
	# Gather any command line arguments
	parser = argparse.ArgumentParser(description='Coordinator parameters')
	parser.add_argument('--process_ID', type=int, default=9, #required=True,
						help='The unique ID for this process')
	args = parser.parse_args()

	# Create an instance of the core coordinator
	coordinator = Coordinator(args.process_ID)
	coordinator.initialise()
	# Let's process some images!
	coordinator.run()
