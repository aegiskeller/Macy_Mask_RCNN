# fit a mask rcnn on the pothole dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import sys,os

import pothole

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
config = pothole.PotholeConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
#model.load_weights(os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5'), by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# load the last trained weights
model.load_weights(model.find_last(), by_name=True)
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=15, layers='heads')