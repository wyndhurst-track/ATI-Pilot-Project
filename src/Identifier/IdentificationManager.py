#!/usr/bin/env python
import os
import sys
sys.path.append("../")

# My libraries
from config import cfg
from Identifier.MetricLearningWrapper import MetricLearningWrapper

class IdentificationManager(object):
	#This class manages the identification pipeline
	def __init__(self, load_weights=True):
		# Initialise our identification method
		if cfg.ID.ID_METHOD == cfg.IDENTIFIERS.METRIC_LEARNING:
			self.__identifier = MetricLearningWrapper()
		else:
			print('Will did not define ClosedSetWrapper() --jING')

	def identifyBatch(self, images):
		return self.__identifier.predictBatch(images)

# Entry method/unit testing method
if __name__ == '__main__':
	pass
