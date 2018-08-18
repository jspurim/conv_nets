import argparse
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.models import load_model
from more_itertools import pairwise


def get_convolutional_activation_layers(model):
    return [current for prev, current in pairwise(model.layers) if
            current.__class__ == Activation and prev.__class__ == Conv2D]


def make_convolutional_filter_models(model):
    first_layer = model.layers[0]
    layers = get_convolutional_activation_layers(model)
    return [Model(input=first_layer.input, output=layer.output)
            for layer in layers]

def output_filters(layer_num, filters, output_folder):
    for i, f in enumerate(filters):
        cv2.imwrite("%s/layer%02d_f%02d.png"%(output_folder,layer_num,i),f*255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description="Process an image using a cnn model and generate mosaic"
      + " image of the output of it's convolutional layers")
    parser.add_argument(
      "-m", "--model", required=True,
      help="path to trained model model")
    parser.add_argument(
      "-i", "--image", required=True,
      help="path to input image")
    parser.add_argument("-o", "--output", help="Output folder", required=True)
    args = vars(parser.parse_args())

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    original_model = load_model(args["model"])
    input_shape = original_model.layers[0].input_shape[1:-1]

    # load the image
    print("[INFO] loading image...")
    image = cv2.imread(args["image"])
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, input_shape)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    print("[INFO] Building truncated models...")
    models = make_convolutional_filter_models(original_model)

    print("[INFO] Computing filters...")
    for i,model in enumerate(make_convolutional_filter_models(original_model)):
        prediction = model.predict(image)
        shape = prediction.shape
        filters = np.rollaxis(prediction.reshape(shape[1:]), 2)
        output_filters(i, filters, args["output"])
