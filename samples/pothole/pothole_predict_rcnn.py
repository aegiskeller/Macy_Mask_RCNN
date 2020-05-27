# evaluate the mask rcnn model on the pothole dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.model import mold_image
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import sys,os

import pothole

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "pothole_cfg"
	# number of classes (background + pothole)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0.90

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=1):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# define subplot
		figure = pyplot.gcf() # get current figure
		figure.set_size_inches(8, 6)
		pyplot.subplot(n_images, 2, i*2+1)
		# turn off axis labels
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(image)
		if i == 0:
			pyplot.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		pyplot.subplot(n_images, 2, i*2+2)
		# turn off axis labels
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(image)
		if i == 0:
			pyplot.title('Predicted')
		ax = pyplot.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	#pyplot.show()
	pyplot.savefig('%s.png' %(dataset), dpi=150)

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
# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, config)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, config)