#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("/home/will/work/1-RA/src")
from tqdm import tqdm

# PyTorch libraries
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# My libraries
from config import cfg
from Utilities.Utility import Utility

"""
Implements basic incremental learning by freezing the last x layers of the network and fine-
tuning on new data. Network is ResNet-101 using pyTorch
"""

class IncrementalLearnerPyTorch(object):
	# Class constructor
	def __init__(self, ):
		"""
		Class attributes
		"""

		# Where to save models
		self.__save_dir = os.path.join(cfg.ID.ID_DIR, "data/models/IL.net")

		# Where to find the datasets
		self.__train_data_dir = os.path.join(cfg.ID.ID_DIR, "data/cows_train")
		self.__test_data_dir = os.path.join(cfg.ID.ID_DIR, "data/cows_test")

		"""
		Class objects
		"""

		"""
		Class setup
		"""

		self.__setup()

	"""
	Public methods
	"""

	def trainNetwork(self):
		train_batch_size = 128
		epochs = 20

		# transform = transforms.Compose(
		# 	[transforms.ToTensor(),
		# 	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		# 										download=True, transform=transform)

		transform = transforms.Compose([	transforms.RandomCrop(224),
											transforms.ToTensor()		])
		data_path = "/home/will/work/1-RA/src/Datasets/data/CowID/"
		trainset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

		train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
												  shuffle=True, num_workers=2)

		print("Commencing training")

		# Start the training loop
		total_iterations = int(epochs * (len(trainset)/train_batch_size))
		pbar = tqdm(total=total_iterations)
		for epoch in range(epochs):
			running_loss = 0.0

			# Train a minibatch
			for i, data in enumerate(train_loader, 0):
				# Get the data and labels
				inputs, labels = data[0].to(self.__device), data[1].to(self.__device)

				# Zero the parameter gradients
				self.__optimiser.zero_grad()

				# Forward + backward + optimise
				outputs = self.__model(inputs)
				loss = self.__loss(outputs, labels)
				loss.backward()
				self.__optimiser.step()

				# Print statistics on this batch (every x minibatches)
				running_loss += loss.item()
				if i % 200 == 199:
					print(f"[{epoch+1}, {i+1}] CE loss: {running_loss/200}")
					running_loss = 0.0

				pbar.update()
		pbar.close()

		print("Finished training!")

		# Save the model to file
		self.__saveModel()

	def testNetwork(self):
		# Load network weights
		self.__loadModel()

		test_batch_size = 4

		# Get the training data
		transform = transforms.Compose(
			[transforms.ToTensor(),
			 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
									   download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
												 shuffle=False, num_workers=2)

		correct = 0
		total = 0
		with torch.no_grad():
			pbar = tqdm(total=int(len(testset)/test_batch_size))
			for data in testloader:
				images, labels = data[0].to(self.__device), data[1].to(self.__device)
				outputs = self.__model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				pbar.update()
			pbar.close()

		print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))

	"""
	(Effectively) private methods
	"""

	# Setup the class for operation
	def __setup(self):
		# Setup the GPU
		assert(torch.cuda.is_available())
		self.__device = torch.device("cuda:0")

		# Get the resnet101 architecture pretrained on imagenet
		self.__model = models.resnet50(pretrained=True)
		# self.__model = models.vgg11()
		# self.__model = Net()

		# Put the model on the GPU
		self.__model.to(self.__device)

		# Define our cross entropy loss function
		self.__loss = nn.CrossEntropyLoss()

		# Define the optimiser (SGD with momentum)
		self.__optimiser = optim.SGD(self.__model.parameters(), lr=0.001, momentum=0.9)

	def __setupData(self):
		pass

	def __loadModel(self):
		self.__model.load_state_dict(torch.load(self.__save_dir))
		print(f"Loaded model from: {self.__save_dir}")

	def __saveModel(self):
		torch.save(self.__model.state_dict(), self.__save_dir)
		print(f"Saved model to: {self.__save_dir}")

	"""
	Getters
	"""

	"""
	Setters
	"""

	"""
	Static methods
	"""

# Very very basic CNN
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# Entry method/unit testing method
if __name__ == '__main__':
	# Create an instance
	identifier = IncrementalLearnerPyTorch()

	# Train the network up
	identifier.trainNetwork()

	# Test the network
	# identifier.testNetwork()