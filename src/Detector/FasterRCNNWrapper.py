import sys
sys.path.append("../")
sys.path.append("D:\\Work\\Other\\detectron2")

# Base libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import numpy as np
import cv2
import time
import random
from google.colab.patches import cv2_imshow
import itertools
import json
import argparse
from tqdm import tqdm

# My libraries
from Utilities.DataUtils import DataUtils
from Visualiser.VisualisationManager import VisualisationManager as VM
from Detector.DarknetWrapper import DarknetWrapper

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import PascalVOCDetectionEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

"""
This class wraps FasterRCNN implemented via detectron2 so that it can be used for inference
and therefore detection
"""

class FasterRCNNWrapper():
	# Class constructor
	def __init__(self, load_weights=True):
		# # Base directory for data/weights/etc. stored for this wrapper
		# self.__base_dir = "/home/will/work/1-RA/src/Detector/data/FasterRCNN"

		# # Path for network configuration
		# self.__network_def = os.path.join(self.__base_dir, "COCO-Detection-models/faster_rcnn_R_50_C4_1x.yaml")

		# # Path for trained weights
		# # self.__weights_loc = os.path.join(self.__base_dir, "trained_for_150000_iters/model_final.pth")
		# self.__weights_loc = os.path.join(self.__base_dir, "fold-0/model_final.pth")

		# TEMPORARY for testing on BP1
		fold = 9
		self.__weights_loc = f"/work/ca0513/models/faster-rcnn/output/fold-{fold}/model_final.pth"
		self.__network_def = f"/work/ca0513/models/detectron2/faster_rcnn_R_50_C4_1x.yaml"

		# If we're supposed to be loading weights from file
		if load_weights:
			config = get_cfg()
			config.merge_from_file(self.__network_def)
			config.DATALOADER.NUM_WORKERS = 2
			config.SOLVER.IMS_PER_BATCH = 8
			config.SOLVER.BASE_LR = 0.00025
			config.MODEL.WEIGHTS = self.__weights_loc
			config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
			config.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cow)
			config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
			self.__predictor = DefaultPredictor(config)

	"""
	Public methods
	"""

	def detectBatch(self, images):
		""" Detect on a batch (list) of images """
		return self.__predictor(images)

	# Detect using faster rcnn on an image
	def detectSingle(self, image):
		# Predict on the loaded image
		return self.__predictor(image)

	"""
	(Effectively) private methods
	"""

	# Simply predict on an image
	def testPrediction():
		im = cv2.imread("./input.jpg")
		cfg = get_cfg()
		cfg.merge_from_file("/home/will/work/other/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
		# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
		cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
		predictor = DefaultPredictor(cfg)
		outputs = predictor(im)
		v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		cv2.imshow("Input image", v.get_image()[:, :, ::-1])
		cv2.waitKey(0)

	@staticmethod
	def getCowDict(filepath):
		# Create a list of object annotations
		lines = [line.rstrip() for line in open(filepath, "r")]

		# The dataset dictionary
		dataset_dicts = []

		# Iterate over each image file path
		pbar = tqdm(total=len(lines))
		for idx, img_path in enumerate(lines):
			# Individual instance
			record = {}

			# Load properties about the image
			img_h, img_w = cv2.imread(img_path).shape[:2]
			record["file_name"] = img_path
			record["image_id"] = idx
			record["height"] = img_h
			record["width"] = img_w

			# Construct the path to the annotation file
			anno_path = os.path.join(os.path.dirname(filepath), "labels-xml", os.path.basename(img_path)[:-4]+".xml")

			# Extract the corresponding annotation for this image
			annotation = DataUtils.readXMLAnnotation(anno_path)

			# Iterate over every object
			objs = []
			for anno in annotation['objects']:
				# Convert the annotation into (x1, y1, x2, y2) from darknet form
				# x1 = int((anno[0] - anno[2]/2) * img_w)
				# y1 = int((anno[1] - anno[3]/2) * img_h)
				# x2 = x1 + int(anno[2] * img_w)
				# y2 = y1 + int(anno[3] * img_h)

				# Create the the object and add it to the list
				obj = {
					"bbox": [anno['x1'], anno['y1'], anno['x2'], anno['y2']],
					"bbox_mode": BoxMode.XYXY_ABS,
					"category_id": 0
				}
				objs.append(obj)

			# Add everything to the list
			record["annotations"] = objs
			dataset_dicts.append(record)

			pbar.update()
		pbar.close()

		return dataset_dicts

	@staticmethod
	def prepareCowDataset(directory, fold):
		for d in ["train", "valid"]:
			# Create the path to the train/valid split file
			filepath = os.path.join(directory, f"0-{d}.txt")
			DatasetCatalog.register("cow_" + d, lambda x=filepath: FasterRCNNWrapper.getCowDict(x))
			MetadataCatalog.get("cow_" + d).set(thing_classes=["cow"])
		cow_metadata = MetadataCatalog.get("cow_train")

		# dataset_dicts = getCowDict(os.path.join(directory, "0train.txt"))
		# for d in random.sample(dataset_dicts, 3):
		# 	img = cv2.imread(d["file_name"])
		# 	visualizer = Visualizer(img[:, :, ::-1], metadata=cow_metadata, scale=0.5)
		# 	vis = visualizer.draw_dataset_dict(d)
		# 	cv2.imshow("test", vis.get_image()[:, :, ::-1])
		# 	cv2.waitKey(0)

		cfg = get_cfg()
		cfg.merge_from_file("/work/ca0513/models/detectron2/faster_rcnn_R_50_C4_1x.yaml")
		# cfg.merge_from_file("/home/will/work/other/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
		cfg.DATASETS.TRAIN = ("cow_train",)
		cfg.DATASETS.TEST = ()
		cfg.DATALOADER.NUM_WORKERS = 2
		# cfg.MODEL.WEIGHTS = "/home/will/work/1-RA/src/Detector/data/models/model_final_721ade.pkl"  # initialize from model zoo
		cfg.MODEL.WEIGHTS = "/work/ca0513/models/detectron2/model_final_721ade.pkl"
		cfg.SOLVER.IMS_PER_BATCH = 8
		cfg.SOLVER.BASE_LR = 0.00025
		cfg.SOLVER.MAX_ITER = 70000    # 300 iterations seems good enough, but you can certainly train longer
		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cow)

		print(f"Output directory = {cfg.OUTPUT_DIR}")

		# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
		cfg.OUTPUT_DIR = f"/work/ca0513/models/faster-rcnn/output/fold-{fold}/"
		os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
		trainer = DefaultTrainer(cfg) 
		trainer.resume_or_load(resume=False)
		trainer.train()

		# cfg = get_cfg()
		# cfg.merge_from_file("/home/will/work/other/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
		# trained_weights_path = os.path.join(cfg.OUTPUT_DIR, "trained_for_150000_iters/model_final.pth")
		# trained_weights_path = os.path.join(cfg.OUTPUT_DIR, "trained_for_1000_iters/model_final.pth")
		# trained_weights_path = os.path.join(cfg.OUTPUT_DIR, "trained_for_300_iters/model_final.pth")
		# assert os.path.exists(trained_weights_path)
		# cfg.MODEL.WEIGHTS = trained_weights_path
		# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
		# cfg.DATASETS.TEST = ("cow_test", )
		# predictor = DefaultPredictor(cfg)

		# testDetector(directory, predictor)

		# dataset_dicts = getCowDict(os.path.join(directory, "0test.txt"))
		# for d in dataset_dicts:    
		# 	im = cv2.imread(d["file_name"])
		# 	outputs = predictor(im)
		# 	print(outputs)
		# 	v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
		# 	v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		# 	cv2.imshow("Input image", v.get_image()[:, :, ::-1])
		# 	cv2.waitKey(0)

		# evaluator = PascalVOCDetectionEvaluator("cow_test")
		# val_loader = build_detection_test_loader(cfg, "cow_test")
		# inference_on_dataset(trainer.model, val_loader, evaluator)

	def testDetector(directory, predictor):
		test_data_dicts = getCowDict(os.path.join(directory, "0test.txt"))

		# Maintain a list of precision values for each tested image
		precisions = []

		# Let's go through each test_file and test
		pbar = tqdm(total=len(test_data_dicts))
		for test_inst in test_data_dicts:
			file = test_inst["file_name"]

			# Load the image
			img = cv2.imread(file)

			# Detect
			start = time.time()
			dets = predictor(img)
			print(f"Detector took {time.time() - start}s on: {file}")

			# Find the corresponding GT via the annotation file
			anno_file = file.replace(".jpg", ".txt")
			assert os.path.isfile(anno_file) and anno_file.endswith(".txt")

			# Extract the annotations
			GT = DarknetWrapper.extractDarknetAnnotations(anno_file)

			# Convert the detections to the format we're after
			outputs = dets["instances"].to("cpu").get_fields()['pred_boxes']
			dets_converted = [x.numpy().astype(int).tolist() for x in outputs]

			# Plot the detection
			_, precision = VM.plotDetections(img, dets_converted, display=False, TP_only=True, GT_boxes=GT, convert_dets=False, save_failures=False)

			precisions.append(precision)
			print(f"Precision: {precision}%, AP: {np.average(precisions)}%")

			pbar.update()
		pbar.close()

	"""
	Testing methods for balloon dataset to get to grips with using detectron
	"""

	def get_balloon_dicts(img_dir):
		json_file = os.path.join(img_dir, "via_region_data.json")
		with open(json_file) as f:
			imgs_anns = json.load(f)

		dataset_dicts = []
		for idx, v in enumerate(imgs_anns.values()):
			record = {}
			
			filename = os.path.join(img_dir, v["filename"])
			height, width = cv2.imread(filename).shape[:2]
			
			record["file_name"] = filename
			record["image_id"] = idx
			record["height"] = height
			record["width"] = width
		  
			annos = v["regions"]
			objs = []
			for _, anno in annos.items():
				assert not anno["region_attributes"]
				anno = anno["shape_attributes"]
				px = anno["all_points_x"]
				py = anno["all_points_y"]
				poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
				poly = list(itertools.chain.from_iterable(poly))

				obj = {
					"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
					"bbox_mode": BoxMode.XYXY_ABS,
					"segmentation": [poly],
					"category_id": 0,
					"iscrowd": 0
				}
				objs.append(obj)
			record["annotations"] = objs
			dataset_dicts.append(record)
		return dataset_dicts

	def trainBalloons():
		balloon_dir = "/home/will/work/1-RA/src/Detector/data/balloon_dataset/balloon/"
		for d in ["train", "val"]:
			DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(balloon_dir + d))
			MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
		balloon_metadata = MetadataCatalog.get("balloon_train")

		cfg = get_cfg()
		cfg.merge_from_file("/home/will/work/other/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
		cfg.DATASETS.TRAIN = ("balloon_train",)
		cfg.DATASETS.TEST = ()
		cfg.DATALOADER.NUM_WORKERS = 2
		cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
		cfg.SOLVER.IMS_PER_BATCH = 2
		cfg.SOLVER.BASE_LR = 0.00025
		cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

		# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
		# trainer = DefaultTrainer(cfg) 
		# trainer.resume_or_load(resume=False)
		# trainer.train()

		cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
		cfg.DATASETS.TEST = ("balloon_val", )
		predictor = DefaultPredictor(cfg)

		dataset_dicts = get_balloon_dicts(balloon_dir + "val")
		for d in random.sample(dataset_dicts, 3):    
			im = cv2.imread(d["file_name"])
			outputs = predictor(im)
			v = Visualizer(im[:, :, ::-1],
						   metadata=balloon_metadata, 
						   scale=0.8, 
						   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
			)
			v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
			# cv2_imshow(v.get_image()[:, :, ::-1])
			cv2.imshow("Prediction", v.get_image()[:, :, ::-1])
			cv2.waitKey(0)

# Entry method/unit testing method
if __name__ == '__main__':
	# Gather command line arguments
	parser = argparse.ArgumentParser(description='Parameters')
	parser.add_argument('--fold', type=int, default=0, help='Which fold to train on')
	parser.add_argument('--dataset_loc', type=str, help='Where to find training data')
	args = parser.parse_args()

	# Just for testing that detectron2 works
	# testPrediction()

	# trainBalloons()

	# Train on own dataset
	# dataset_loc = "/home/will/work/1-RA/src/Detector/data/CEADetection/"
	dataset_dict = FasterRCNNWrapper.prepareCowDataset(args.dataset_loc, args.fold)