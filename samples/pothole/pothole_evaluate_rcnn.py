# evaluate the mask rcnn model on the pothole dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import sys,os

import pothole

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "pothole_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

# prepare train set
ROOT_DIR="../../"
config = pothole.PotholeConfig()
POTHOLE_DIR = os.path.join(ROOT_DIR, "datasets/pothole")
print(POTHOLE_DIR)
train_set = pothole.PotholeDataset()
train_set.load_pothole(POTHOLE_DIR, "train")
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = pothole.PotholeDataset()
test_set.load_pothole(POTHOLE_DIR, "val")
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = PredictionConfig()
config.display()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=config)
# load model weights
model.load_weights('mask_rcnn_pothole_0005.h5', by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, config)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, config)
print("Test mAP: %.3f" % test_mAP)