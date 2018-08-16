# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import lenet
from imutils import paths
import numpy as np
import argparse
import random
import cv2
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")


def load_dataset(dataset_path):
    img_paths = list(paths.list_images(dataset_path))
    random.shuffle(img_paths)
    labels = [x[len(dataset_path):].split("/")[0] for x in img_paths]
    imgs = [cv2.imread(x) for x in img_paths]
    return (imgs, labels)



def train_model(model, dataset, output, epochs, init_lr, batch_size):
    print("[INFO] loading dataset...")
    data, labels = load_dataset(dataset)
    labels_map = {label:i for i, label in enumerate(sorted(list(set(labels))))}
    inv_labels_map = {v:k for k,v in labels_map.iteritems()}
    labels = [labels_map[label] for label in labels]

    print("[INFO] found %s classes" % len(labels_map))
    for k,v in inv_labels_map.iteritems():
        print("\t%s: %s" % (k,v))

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.1,
        horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

    # initialize the model
    print("[INFO] compiling model...")
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
        epochs=epochs, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(output)
    print("[INFO] serializing training stats...")
    f = open(output+".his", "w")
    for stat in ["acc", "loss", "val_acc", "val_loss"]:
        f.write(stat +": "+ " ".join(map(str,H.history[stat]))+"\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a the classifier on the input dataset.")

    # General arguments
    parser.add_argument("-d", "--dataset", help="Path to the dataset folder. Subfolder names are used for the classes.")
    parser.add_argument("-m", "--model", help="Path to the output model.")
    parser.add_argument("-e", "--epochs", help="Number of epochs to train.", type=int, default=100)
    parser.add_argument("-b", "--batch-size", dest="batch_size", help="Number of training cases per batch", type=int, default=32)
    parser.add_argument("--learning-rate", dest="learning_rate", help="Initial learning rate", type=float, default=1e-4)
    parser.add_argument("--width", help="Width of the input images.",type=int, default=300)
    parser.add_argument("--height", help="Height of the input images.",type=int, default=300)
    parser.add_argument("--depth", help="Number of channels of the input images.",type=int, default=3)
    parser.add_argument("--classes", help="Number of channels of the input images.",type=int, default=2)
    parser.add_argument("-s", "--random-seed", help="Seed for the random number generator.",type=int, default=42)

    subparsers = parser.add_subparsers(title="Networks")
    lenet.attach_parser(subparsers)

    args = parser.parse_args()
    random.seed(args.random_seed)

    model = args.build_model(args)
    train_model(model, args.dataset, args.model, args.epochs, args.learning_rate, args.batch_size)
