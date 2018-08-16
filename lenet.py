# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K


def build(width, height, depth, classes, cl1, cl2, dl1, dl2):
    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)
    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(cl1, (5, 5), padding="same",
    	input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(cl2, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(dl1))
    model.add(Activation("relu"))
    # second set of FC => RELU layers
    model.add(Dense(dl2))
    model.add(Activation("relu"))
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model



def attach_parser(subparsers, dest="build_model"):
    """
    Attaches a subparser to the passed parsed with network specific parameters.
    Assumes parent parser provides the width, height, depth and classes parameters.
    Will attach a function to create the network model to the parsed arguments.
    """

    def init_model(values):
        model = build(values.width, values.height, values.depth, values.classes,
            values.cl1, values.cl2, values.dl1, values.dl2)
        return model

    parser = subparsers.add_parser("lenet", description="Use LeNet network.")
    parser.add_argument("--dl1", type=int, default=512)
    parser.add_argument("--dl2", type=int, default=512)
    parser.add_argument("--cl1", type=int, default=32)
    parser.add_argument("--cl2", type=int, default=64)
    parser.set_defaults(**{dest:init_model})
