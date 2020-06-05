# evaluate the mask rcnn model on the airbus dataset
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

import airbus

# define classes that the model knowns about
class_names = ['BG', 'airbus']

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "airbus_cfg"
	# number of classes (background + airbus)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0.90

# prepare train set
ROOT_DIR="../../"
config = airbus.AirbusConfig()
AIRBUS_DIR = os.path.join(ROOT_DIR, "datasets/airbus-ship-detection")
print(AIRBUS_DIR)
train_set = airbus.AirbusDataset()
train_set.load_airbus(AIRBUS_DIR, "train")
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = airbus.AirbusDataset()
test_set.load_airbus(AIRBUS_DIR, "val")
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = PredictionConfig()
config.display()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=config)
# load model weights
model_path = 'mask_rcnn_airbus_0050.h5'
model.load_weights(model_path, by_name=True)
# load photograph
img = load_img('../../datasets/airbus-ship-detection/test_v2/0010551d9.jpg')
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