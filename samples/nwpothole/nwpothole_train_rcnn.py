# fit a mask rcnn on the nwpothole dataset
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import sys,os
import argparse

import nwpothole

parser = argparse.ArgumentParser()
parser.add_argument("--init", help='initialise from mscoco? Boolean', type=bool)
args = parser.parse_args()
# prepare train set
ROOT_DIR="../../"
config = nwpothole.NwpotholeConfig()
NWPOTHOLE_DIR = os.path.join(ROOT_DIR, "datasets/nwpothole")
print(NWPOTHOLE_DIR)
train_set = nwpothole.NwpotholeDataset()
train_set.load_nwpothole(NWPOTHOLE_DIR, "train")
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = nwpothole.NwpotholeDataset()
test_set.load_nwpothole(NWPOTHOLE_DIR, "val")
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = nwpothole.NwpotholeConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
if args.init:
    print('Training from msCOCO')
    model.load_weights(os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5'), by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# load the last trained weights
else:
    print('Training from last weights')
    model.load_weights(model.find_last(), by_name=True)
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')