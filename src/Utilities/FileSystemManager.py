# Core libraries
import os
import sys
sys.path.append("../")
from datetime import datetime
from datetime import timedelta  

# My libraries
from config import cfg
from Utilities.DataUtils import DataUtils

class FileSystemManager(object):
	"""
	Class descriptor

	This class, and indeed the entire codebase currently expects filename in the form:
	2020-09-30-10-56-24_CID-001.mp4

	Once it has been moved and locked by a process, it will take the name
	2020-09-30-10-56-24_CID-001_PID-001.mp4
	"""

	def __init__(self, process_ID, logging_manager):
		"""
		Class constructor
		"""

		# Unique identifier for this process
		self.__process_ID = process_ID

		# Reference to logging manager instance
		self.__logging_manager = logging_manager

		# Path to the workspace
		self.__workspace_path = cfg.FSM.WORKSPACE

		# Video extension to expect
		self.__video_extension = cfg.FSM.VIDEO_EXTENSION

		# Path to the folder full of videos ready to be processed
		self.__to_be_processed_path = os.path.join(self.__workspace_path, cfg.FSM.FOLDER_2)

		# Path to the temporary folder containing videos that are currently being processed
		self.__processing_path = os.path.join(self.__workspace_path, cfg.FSM.FOLDER_3)

		# Path to the folder full of videos that have been processed
		self.__processed_path = os.path.join(self.__workspace_path, cfg.FSM.FOLDER_4)

		# Filepath of file we're currently processing
		self.__currently_processing = None

		# Check all the folders exist
		print(self.__workspace_path)
		assert os.path.exists(self.__workspace_path)
		assert os.path.exists(self.__to_be_processed_path)
		assert os.path.exists(self.__processing_path)
		assert os.path.exists(self.__processed_path)

	"""
	Public methods
	"""	

	def checkForUnprocessed(self, debug_dont_move=False):
		""" Check for unprocessed videos

		if debug_dont_move is true, dont move or rename the file so that the file doesn't need to be moved and renamed
		each time the code is tested

		"""

		# Check for new videos (sorted by timestamp; oldest item first)
		files = DataUtils.allFilesAtDirWithExt(self.__to_be_processed_path, self.__video_extension)

		# There is at least one video to be processed
		if len(files) > 0:
			# Reserve it by moving it to the processing folder and renaming it
			new_filename = self.__addPID(os.path.basename(files[0]))
			new_path = os.path.join(self.__processing_path, new_filename)

			# Another process might have taken the file before we were able to reserve it, just return None
			try:
				# Only rename the file if we're not in debug mode
				if not debug_dont_move:
					# Once we've done this, we have the lock on the file
					os.rename(files[0], new_path)
				else:
					# Just reassign the old path as the new one
					new_path = files[0]

				# Temporarily store the path to this file we're processing
				self.__currently_processing = new_path

				# Split just the filename by underscores as a delimter
				split_filename = new_filename.split("_")

				# Extract the datetime
				timestamp = datetime.strptime(split_filename[0], "%Y-%m-%dT%H-%M-%S")
				offset = int(split_filename[2])
				timestamp = timestamp + timedelta(minutes=offset)

				# The camera ID is the second part of the second element
				camera_ID = int(split_filename[1].split("-")[1])

				# Report that we have locked this file to this process
				self.__logging_manager.logInfo(__file__, f"locked file now at: {new_path}")

				# Return the path to this file along with the camera ID and datetime for this file
				return True, new_path, camera_ID, timestamp

			except OSError as e:
				return False, None, None, None

		return False, None, None, None

	def finishedProcessing(self, debug_dont_move=False):
		""" Moves a processed file to the last folder """

		# Don't bother if we're in debug mode
		if not debug_dont_move:
			# Make sure current processing is correct
			assert os.path.exists(self.__currently_processing)

			# Simply move this file to the last "processed" folder, retaining its name
			new_path = os.path.join(self.__processed_path, os.path.basename(self.__currently_processing))
			os.rename(self.__currently_processing, new_path)

			# Report that we've finished with this file and it has been moved elsewhere
			self.__logging_manager.logInfo(__file__, f"processed file moved to: {new_path}")

		# Empty the filepath to the file being processed
		self.__currently_processing = None

	def checkProcessing(self):
		""" 
		Checks whether any files are present in the processing folder that haven't been fully processed.
		This might have occurred if a processing instance crashed halfway through processing a file.
		If there are any, it will move the file back to the to_be_processed folder and remove the process ID tag.
		Note, to solve concurrency issues, it will only do this for files that had the same process ID as this
		"""

		# Check for videos in the processing_in_progress folder
		processing_files = DataUtils.allFilesAtDirWithExt(self.__processing_path, self.__video_extension)

		# Only keep files that have the same process ID as this instance
		processing_files = [x for x in processing_files if f"PID-{str(self.__process_ID).zfill(3)}" in x]

		# Are there any files there?
		if len(processing_files) > 0:
			# Report that this is the case
			log_str = f"Found {len(processing_files)} files that were midway"
			log_str += f"through processing, returning them to be processed."
			self.__logging_manager.logInfo(__file__, log_str)

			# Iterate through them
			for original_filepath in processing_files:
				# Strip the process ID from the filename
				new_filename = self.__stripPID(os.path.basename(original_filepath))

				# Create the full destination filepath
				new_path = os.path.join(self.__to_be_processed_path, new_filename)

				# Move the file back to be processed
				os.rename(original_filepath, new_path)

				# Report that we've moved this file back
				self.__logging_manager.logInfo(__file__, f"midprocessed file moved back to: {new_path}")

	"""
	(Effectively) private methods
	"""

	def __addPID(self, filename):
		""" Append the process ID to a given filename """

		# Strip the file extension and the full stop
		filename = filename[:-len(self.__video_extension)-1]

		# Add the process ID and file extension back
		new_filename = filename + f"_PID-{str(self.__process_ID).zfill(3)}.{self.__video_extension}"

		return new_filename

	def __stripPID(self, filename):
		""" Remove the process ID from a given filename 
		e.g. 			2020-02-05-13-27-01_CID-001_PID-001.avi
		would become:	2020-02-05-13-27-01_CID-001.avi
		"""

		# Split the string by underscores
		split_filename = filename.split("_")

		# Join the first two elements
		new_filename = "_".join(split_filename[:2])

		# Append the file extension
		new_filename += f".{self.__video_extension}"

		return new_filename

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
	pass
	