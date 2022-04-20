# Core libraries
import os
import sys, cv2
import pandas as pd
sys.path.append("../")

# My libraries
from config import cfg
from Utilities.DataUtils import DataUtils
from Identifier.IdentificationManager import IdentificationManager


def allFiles(sum, directory, file_extension, full_path=True):

	files= [os.path.join(directory, x) for x in sorted(os.listdir(os.path.join(sum,directory))) if x.endswith(file_extension) and x[12]=='0']

	mmm = []
	for image in files:
		img = cv2.imread(os.path.join(sum, image))
		mmm.append(img)

	batched = DataUtils.chunks(mmm, cfg.ID.BATCH_SIZE)
	# Get the outputs for each batch
	outputs = [IdentificationManager().identifyBatch(batch) for batch in batched]
	outputs = [item for sublist in outputs for item in sublist]  # to a 1D list

	return files, outputs

	# Entry method/unit testing method
if __name__ == '__main__':
	#read images from files
	folder_path = '/home/will/Downloads/Crop_h150/'
	save_path = '/home/will/Desktop/'
	pair_list = []

	for x_f in sorted(os.listdir(folder_path)):
		image, label = allFiles(folder_path, x_f, ".jpg")
		for i,j in zip(image,label):
			pair_list.append([i, j])
		print(label)

	data1 = pd.DataFrame(pair_list)
	data1.to_csv(save_path + 'label.csv')
