# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True,
	help="path to trained model model")
parser.add_argument("-i", "--image", required=True,
	help="path to input image")
parser.add_argument("-l", "--labels", nargs="+")
args = vars(parser.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (300, 300))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
prediction = model.predict(image)[0]
for i,p in enumerate(prediction):
    print "%s: %0.2f%%" % (args["labels"][i], p*100)
