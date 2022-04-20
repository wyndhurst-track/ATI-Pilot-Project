# Core libraries
import os
import sys
sys.path.append(os.path.abspath("../"))
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

# My libraries
from Utilities.DataUtils import DataUtils

# Class for visualising data from model training (e.g. loss, accuracy vs steps)

class TrainingVisualiser(object):
	# Class constructor
	def __init__(self):
		pass

	"""
	Public methods
	"""

	"""
	(Effectively) private methods
	"""

	"""
	Staticmethods
	"""

	@staticmethod
	def visualiseTrainingGraph(epochs, train_loss, val_loss, train_acc, val_acc):
		fig, ax1 = plt.subplots()
		colour1 = 'tab:blue'
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Accuracy', color=colour1)
		ax1.set_xlim((0, np.max(epochs)))
		ax1.set_ylim((0.,1.))
		ax1.plot(epochs, val_acc, color=colour1)

		ax2 = ax1.twinx()
		colour2 = 'tab:red'
		ax2.set_ylabel('Loss', color=colour2)
		ax2.set_ylim((0., 3.))
		ax2.plot(epochs, val_loss, color=colour2)

		plt.tight_layout()
		plt.show()
		# plt.savefig()

	@staticmethod
	def visualiseFromPickleFile(file_path):
		print(f"Loading data from file: {file_path}")

		# Load the data
		with open(file_path, 'rb') as handle:
			data = pickle.load(handle)

		# Add epoch data
		epochs = np.arange(0, 100)

		# Extract arrays from this
		train_loss = np.array(data['loss']['train'])
		val_loss = np.array(data['loss']['val'])
		train_acc = np.array(data['acc']['train'])
		val_acc = np.array(data['acc']['val'])

		# Render a graph from this
		graph = TrainingVisualiser.visualiseTrainingGraph(epochs, train_loss, val_loss, train_acc, val_acc)

	@staticmethod
	def visualiseFromNPZFile(file_path):
		print(f"Loading data from file: {file_path}")

		with np.load(file_path) as data:
			train_loss = data['losses_mean']
			train_steps = data['loss_steps']
			val_acc = data['accuracies']/100
			val_steps = data['accuracy_steps']

		print(f"Best accuracy = {np.max(val_acc)}")

		if val_steps.shape[0] == 0:
			step_size = round(float(np.max(train_steps)) /	val_acc.shape[0])
			val_steps = np.arange(0, np.max(train_steps), step_size)

		max_steps = max(np.max(val_steps), np.max(train_steps))
		print(val_acc)

		fig, ax1 = plt.subplots()
		colour1 = 'tab:blue'
		ax1.set_xlabel('Steps')
		ax1.set_ylabel('Accuracy', color=colour1)
		ax1.set_xlim((0, max_steps))
		ax1.set_ylim((0.,1.))
		ax1.plot(val_steps, val_acc, color=colour1)

		ax2 = ax1.twinx()
		colour2 = 'tab:red'
		ax2.set_ylabel('Loss', color=colour2)
		# ax2.set_ylim((0., 3.))
		ax2.plot(train_steps, train_loss, color=colour2)

		plt.tight_layout()
		plt.show()

	@staticmethod
	def visualiseKGridSearch():
		file_path = "D:\\Work\\ATI-Pilot-Project\\src\\Identifier\\MetricLearning\\output\\k_grid_search.json"

		with open(file_path, 'r') as handle:
			data = json.load(handle)

		X = np.array(list(data.keys())).astype(int)
		Y = np.array([data[x] for x in data.keys()])

		plt.figure()
		plt.plot(X, Y)
		plt.ylabel("Accuracy (%)")
		plt.xlabel("k")
		plt.xlim((0,np.max(X)))
		plt.tight_layout()
		# plt.show()
		plt.savefig("k-grid-search.pdf")

# Entry method/unit testing method
if __name__ == '__main__':
	# Root dir
	# root_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Results\\ClosedSet\\RGBD"
	# files = DataUtils.allFilesAtDirWithExt(root_dir, ".pkl")
	# for file in files:
	# 	if "_data_logs.pkl" in file:
	# 		TrainingVisualiser.visualiseFromPickleFile(file)

	# root_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Results\\LatentSpace"
	# img_type = "RGBD"
	# fold = 9
	# file_path = os.path.join(root_dir, img_type, f"fold_{fold}", "logs.npz")
	# TrainingVisualiser.visualiseFromNPZFile(file_path)

	TrainingVisualiser.visualiseKGridSearch()