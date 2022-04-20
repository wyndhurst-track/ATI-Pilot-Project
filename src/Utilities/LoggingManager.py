#!/usr/bin/env python

# Core libraries
import os
import sys
import logging
import datetime

# My libraries
sys.path.append("../")
from config import cfg

class LoggingManager(object):
	"""
	Utilises python logging class - https://docs.python.org/2/howto/logging.html
	"""

	def __init__(self, process_ID, to_console=False):
		""" Class constructor """

		# The base directory where logs are stored for this logger
		self.__log_dir = os.path.join(cfg.GEN.BASE_DIR, f"logs/process_{process_ID}")

		# The path to the actual log file
		todayDate = datetime.datetime.now()
		logName = todayDate.strftime("%Y-%m-%d") + ".log"
		self.__log_file = os.path.join(self.__log_dir, logName)

		# The process ID this logging instance represents
		self.__process_ID = process_ID

		# Whether to log to the console or to file
		self.__to_console = to_console

		"""
		Class objects
		"""

		"""
		Class setup
		"""
		if not self.__to_console:
			# Create the log folder if it doesn't already exist
			os.makedirs(self.__log_dir, exist_ok=True)

			# Point towards the logging file
			logging.basicConfig(filename=self.__log_file, level=logging.DEBUG)

	"""
	Public methods
	"""

	# Detailed information, typically of interest only when diagnosing problems.
	def logDebug(self, file, msg):
		self.__log(logging.debug, file, msg)

	# Confirmation that things are working as expected.
	def logInfo(self, file, msg):
		self.__log(logging.info, file, msg)

	# An indication that something unexpected happened, or indicative of some problem in the near 
	# future (e.g. ‘disk space low’). The software is still working as expected.
	def logWarning(self, file, msg):
		self.__log(logging.warning, file, msg)

	# Due to a more serious problem, the software has not been able to perform some function.
	def logError(self, file, msg):
		self.__log(logging.error, file, msg)

	# A serious error, indicating that the program itself may be unable to continue running.
	def logCritical(self, file, msg):
		self.__log(logging.critical, file, msg)

	"""
	(Effectively) private methods
	"""

	def __log(self, log_fn, file, msg):
		# Create the string we want to log
		log_str = f" {datetime.datetime.now()} - PID_{str(self.__process_ID).zfill(3)} - {file} - {msg}"

		if self.__to_console:
			print(log_str)
		else:
			log_fn(log_str)

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
	# Create an instance, log some dummy messages
	LM = LoggingManager(0)
	LM.logDebug("Test.py", "This is a test")
	LM.logInfo("Test.py", "This is a test")
	LM.logWarning("Test.py", "This is a test")
	LM.logError("Test.py", "This is a test")
	LM.logCritical("Test.py", "This is a test")
