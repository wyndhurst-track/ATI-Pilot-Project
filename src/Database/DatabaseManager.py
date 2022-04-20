#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import csv
import datetime

# import mysql.connector

# My libraries
from config import cfg

class DatabaseManager(object):
	"""
	This class interacts with the database for retrieving video and writing back results

	Could potentially use python-sql: https://pypi.org/project/python-sql/ which wraps queries
	in a nice pythonic way?
	"""

	def __init__(self, process_ID, logging_manager, to_csv=False):
		""" Class constructor """

		# The process ID this instance belongs to
		self.__process_ID = process_ID

		# Reference to logging manager instance
		self.__logging_manager = logging_manager

		# Whether to just log to a csv file (for debugging when using a SINGLE process)
		self.__to_csv = to_csv

		# Are we just saving to CSV?
		if self.__to_csv:
			# Path to find where the CSV file is stored
			self.__base_dir = cfg.DAT.CSV_PATH

			# Path to the CSV file
			self.__csv_filepath = os.path.join(self.__base_dir, cfg.DAT.CSV_FILE)

			# If the file doesn't already exist, create it and add the first row
			if not os.path.exists(self.__csv_filepath):
				with open(self.__csv_filepath, 'w', encoding="utf8") as file:
					csv_writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
					csv_writer.writerow(('cam_ID', 'process_ID', 'cow_ID', 'cx', 'cy', 'w', 'h', 'angle', 'score', 'datetime', 'frame_ID', 'filename'))
		else:
			# Setup the connection to the sql server
			self.__db = mysql.connector.connect(
				host="localhost",
				user="user",
				passwd="password"
			)
			self.__cursor = self.__db.cursor()

	"""
	Public methods
	"""

	def record(self, camera_ID, obj, datetime, frame_ID, filename):
		""" Record the observation of a cow to the database """

		# Get the parameters to record
		cx, cy, w, h, angle = obj.getBox()
		score = obj.getScore()
		cow_ID = obj.getID()
		self.__logging_manager.logInfo(__file__, f"Record to database: " + str(cow_ID) + " at " + str(datetime))

		# Are we just saving to CSV?
		if self.__to_csv:
			# First we need to check if this entry is already in the database (don't bother adding it if it is)
			# WARNING: may significantly slow down the program for larger databases
			duplicate = False
			with open(self.__csv_filepath, newline='') as file:
				reader = csv.reader(file, delimiter=";", quotechar='"')
				for row in reader:
					if 	row[0] == str(camera_ID) and row[1] == str(self.__process_ID) and row[2] == str(cow_ID) and \
						row[9] == str(datetime) and row[10] == str(frame_ID) and row[11] == filename:
						# This is a duplicate
						duplicate = True

						# Report that this has occurred
						self.__logging_manager.logInfo(__file__, f"Skipping duplicate entry for: {filename}")

			# Open the file in append mode
			with open(self.__csv_filepath, 'a') as file:
				# This isn't a duplicate, proceed to add it
				if not duplicate:
					# Create the writer object
					csv_writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
					
					# Write the row out
					csv_writer.writerow((camera_ID, self.__process_ID, cow_ID, cx, cy, w, h, angle, score, datetime, frame_ID, filename))

		# Actually write to the database
		else:
			# TODO: check whether this is a duplicate entry (is this handled by SQL automatically?)

			# Construct the insertion query
			sql = f" INSERT INTO data (cam_ID, process_ID, cow_ID, cx, cy, w, h, angle, score, datetime, frame_ID, filename) \
					 VALUES (	{cam_ID}, \
					 			{self.__process_ID}, \
					 			{cow_ID}, \
					 			{cx}, {cy}, {w}, {h}, {angle}, {score}, \
					 			{datetime}, \
					 			{frame_ID}, \
					 			{filename})"

			# Actually run it
			ret = self.__executeQuery(sql)

	def close(self):
		""" Close database connections cleanly """

		if self.__to_csv:
			pass
		else:
			pass

	"""
	(Effectively) private methods
	"""

	# Execute a SQL query
	def __executeQuery(self, query_str):
		self.__cursor.execute(query_str)
		result = self.__cursor.fetchall()
		return result

	# UNTESTED: Retrieves the closest timestamped frame for a given camera ID
	def retrieveSample(self, camera_ID, timestamp):
		sql = " SELECT * \
				FROM data \
				WHERE cam_ID = {} AND timestamp = {}"\
				.format(camera_ID, timestamp)

		return self.__executeQuery(sql)

	# UNTESTED: Retrieves all samples for a given camera ID between two timestamps
	def retrieveSamplesInWindow(self, camera_ID, start_timestamp, end_timestamp):
		sql = "	SELECT * \
				FROM data \
				WHERE timestamp BETWEEN {} AND {}"\
				.format(start_timestamp, end_timestamp)

		return self.__executeQuery(sql)

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
	dm = DatabaseManager(0, to_csv=True)
	