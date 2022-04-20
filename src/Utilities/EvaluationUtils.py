#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import json
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# My libraries
from Utilities.Utility import Utility
from Utilities.DataUtils import DataUtils

"""
Class for methods to do with evaluating the performance of a model
"""

class EvaluationUtils:
	
	"""
	Static methods
	"""

	# Compute mAP - based on https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
	@staticmethod
	def VOC_evaluate(GTs, preds, ov_thresh=0.5, plot_pr=False):
		# Determine the total number of ground truth annotations
		n_gt = 0
		for filename in GTs.keys(): n_gt += len(GTs[filename]['objects'])

		# Determine the total number of predictions
		n_preds = 0
		for pred in preds: n_preds += 1

		# Sort the predictions by descending confidence
		confidence = np.array([float(x['conf']) for x in preds])
		sorted_ind = np.argsort(-confidence)
		preds = [preds[x] for x in sorted_ind]
		assert len(preds) == n_preds

		# Store info about true positive and false positive predictions
		tp = np.zeros(n_preds)
		fp = np.zeros(n_preds)

		print(f"Number of ground truth objects = {n_gt}")
		print(f"Number of predicted objects = {n_preds}")

		# Loop over every test image
		for i in tqdm(range(n_preds)):
			# Extract the prediction
			pred = preds[i]

			# Extract the image filename this prediction is from
			filename = pred['filename']

			# Extract the ground truth boxes for this image
			gt = GTs[filename]['objects']
			
			# Convert the prediction box
			boxA = [pred['x1'], pred['y1'], pred['x2'], pred['y2']]

			# Save the IoU for the maximal match
			max_IoU = -np.inf
			max_gtbox = None

			# Loop through each ground truth box and determine the IoU with the current
			# prediction
			for gt_box in gt:
				# Convert the ground truth box
				boxB = [gt_box['x1'], gt_box['y1'], gt_box['x2'], gt_box['y2']]

				# Compute the IoU
				IoU = Utility.bboxIntersectionOverUnion(boxA, boxB)

				# Should we update our record of the best IoU and gt_box
				if IoU > max_IoU: 
					max_IoU = IoU
					max_gtbox = gt_box

			# Is the best IoU greater than our threshold? If so, mark it as a TP
			# otherwise FP
			if max_IoU > ov_thresh: 
				tp[i] = 1.

				# Remove the gt box from the list so it doesn't get marked as a TP
				# more than once
				GTs[filename]['objects'].remove(max_gtbox)
			else: fp[i] = 1.

		# Compute precision and recall
		fp = np.cumsum(fp)
		tp = np.cumsum(tp)
		rec = tp / float(n_gt)
		prec = tp / (tp + fp)

		# Append sentinel values at the end
		mrec = np.concatenate(([0.], rec, [1.]))
		mprec = np.concatenate(([0.], prec, [0.]))

		# Compute the precision envelope
		for i in range(mprec.size - 1, 0, -1):
			mprec[i - 1] = np.maximum(mprec[i - 1], mprec[i])

		# To calculate the area under PR curve, look for points where X axis 
		# (recall) changes value
		i = np.where(mrec[1:] != mrec[:-1])[0]

		# And sum delta recall * prec
		mAP = np.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1])

		# Let's plot the precision-recall curve
		if plot_pr:
			plt.plot(mrec, mprec)
			plt.xlabel('Recall')
			plt.ylabel('Precision')
			plt.xlim((0,1))
			plt.ylim((0,1))
			plt.tight_layout()
			plt.show()

		return mAP

	# Affirm that Jing's mAP results are the same as mine
	@staticmethod
	def affirm():
		# Which fold are we affirming
		fold = 9

		# Above which image ID do we ignore (because it is synthetic)
		ignore_id = 3707

		# Load the train/test splits
		splits_fp = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\splits\\10-fold-CV.json"
		with open(splits_fp) as json_file:
			splits = json.load(json_file)

		# Where to find ground truth annotations
		gt_fp = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\VOC"

		# Where to find the predictions
		preds_base = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Papers\\Own\\CEiA\\results\\detection\\retinaNet"
		preds_file = os.path.join(preds_base, f"{str(fold).zfill(2)}.json")
		with open(preds_file) as json_file:
			preds_json = json.load(json_file)

		# Maintain a list of ground truth and predictions
		GTs = {}
		preds = []

		# Loop through jing's json file
		for pred in preds_json:
			# Extract coordinates
			x1 = pred['bbox'][0]
			y1 = pred['bbox'][1]
			w = pred['bbox'][2]
			h = pred['bbox'][3]

			# Convert to two points format
			x2 = x1 + w
			y2 = y1 + h

			preds.append({	'x1': x1, 
							'y1': y1, 
							'x2': x2, 
							'y2': y2, 
							'conf': pred['score'],
							'filename': pred['file_name']})

		# Loop through each test image
		for test_img_fp in splits[str(fold)]['test']:
			# Make sure the image is non-synthetic
			if int(test_img_fp[:-4]) <= ignore_id:
				# Construct the path to the annotation file
				anno_path = os.path.join(gt_fp, test_img_fp[:-4]+".xml")

				# Load the ground truth for this test image
				annotation = DataUtils.readXMLAnnotation(anno_path)

				filename = os.path.basename(anno_path)
				filename = filename.replace(".xml", ".jpg")

				# print(filename)
				# input()

				# Add this to the dict
				# GTs[annotation['filename']] = annotation
				GTs[annotation['filename']] = annotation

		# Compute mAP
		mAP = EvaluationUtils.VOC_evaluate(GTs, preds)
		print(f"mAP = {mAP}")

	# Affirm Jing's results in the second format she provided
	@staticmethod
	def affirm2(fold):
		# Root directory for Jing's results
		root_dir = "C:\\Users\\ca051\\Downloads\\files_for_will_to_calculate_mAP\\files_for_will_to_calculate_mAP\\best"

		# Root directory for the dataset
		dataset_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection"

		# Where to find ground truth labels and splits
		gt_dir = os.path.join(dataset_dir, "labels-xml")
		splits_fp = os.path.join(dataset_dir, "10-fold-CV.json")

		# Maintain a list of ground truth and predictions
		GTs = {}
		preds = []

		# Load the splits into memory
		with open(splits_fp) as handle:
			splits = json.load(handle)

		# Loop through the predictions
		for pred_fp in DataUtils.allFilesAtDirWithExt(os.path.join(root_dir, str(fold)), ".xml"):
			# Load the annotation
			pred = DataUtils.readRotatedXMLAnnotation(pred_fp)

			# Loop through each object
			for obj in pred['objects']:
				# Extract the coordinates
				xc = obj['cx']
				yc = obj['cy']
				w = obj['w']
				h = obj['h']

				# Convert to (x1,y1),(x2,y2)
				x1 = xc - w/2
				y1 = yc - h/2
				x2 = xc + w/2
				y2 = yc + h/2

				# Add to the list of annotations
				preds.append({	'x1': x1, 
								'y1': y1, 
								'x2': x2, 
								'y2': y2, 
								'conf': obj['score'], 
								'filename': pred['image_filename']	})

		# Loop through the testing images
		for test_img_fp in range(splits[str(fold)]['test']):
			# Get the corresponding ground truth label for this image
			anno_path = os.path.join(gt_dir, test_img_fp[:-4]+".xml")
			annotation = DataUtils.readXMLAnnotation(anno_path)

			# Add this to the dict
			GT[annotation['filename']] = annotation

		# Compute mAP
		mAP = EvaluationUtils.VOC_evaluate(GTs, preds)
		print(f"mAP = {mAP} for fold = {fold}")


# Entry method/unit testing method
if __name__ == '__main__':
	# Let's evaluate a saved file
	# filepath = "/home/will/work/1-RA/src/Detector/fold-0_to_be_evaluated.pickle"
	# filepath = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Papers\\Own\\CEiA\\results\\detection\\Faster-RCNN\\fold-1-to_be_evaluated.pickle"
	# with open(filepath, 'rb') as handle:
	# 	a = pickle.load(handle)
	# mAP = EvaluationUtils.VOC_evaluate(a['num_images'], a['GTs'], a['preds'])
	# print(f"mAP = {mAP}")

	# # Let's evaluate a saved json file
	# fold = 9
	# # filepath = f"/home/ca0513/ATI-Pilot-Project/src/Detector/fold-{fold}_to_be_evaluated.json"
	# # filepath = f"D:\\Work\\ATI-Pilot-Project\\src\\Detector\\fold-{fold}-to_be_evaluated.json"
	# filepath = f"D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Papers\\Own\\CEiA\\results\\detection\\Faster-RCNN\\fold-{fold}_to_be_evaluated.json"
	# with open(filepath, 'r') as handle:
	# 	a = json.load(handle)
	# mAP = EvaluationUtils.VOC_evaluate(a['GT'], a['preds'])
	# print(f"mAP = {mAP}")

	# Temporary function to confirm Jing's mAP results are the same as mine
	for k in range(10):
		EvaluationUtils.affirm2(k)