# evaluate the mask rcnn model on the nwpothole dataset
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.model import mold_image
from mrcnn.visualize import display_instances
import sys,os

import nwpothole

# define classes that the model knowns about
class_names = ['BG', 'nwpothole']

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "nwpothole_cfg"
	# number of classes (background + nwpothole)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0.90

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
config = PredictionConfig()
config.display()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=config)
# load model weights
model_path = 'mask_rcnn_nwpothole_0092.h5'
model.load_weights(model_path, by_name=True)
# load photograph
img = load_img('../../datasets/nwpothole/frame240.jpg')
img = img_to_array(img)
# convert pixel values (e.g. center)
scaled_image = mold_image(img, config)
# convert image into one sample
sample = expand_dims(scaled_image, 0)
# make prediction
results = model.detect(sample, verbose=0)
# get dictionary for first prediction
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])