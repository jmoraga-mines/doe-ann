"""

  DOE Project : 
  Task        : 
  File        : 
  
    This is the main program to create, train and use an ANN to classify
  regions based on geothermal potential.

"""


from __future__ import print_function
# import the necessary packages
import argparse
from imutils import paths
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import AveragePooling2D, Input, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam, Nadam, Adadelta, Adagrad, Adamax, SGD
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
#from tensorflow.compat.v2.keras.utils import multi_gpu_model


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import os
import os.path as path
import sys
import pickle
import random
import skimage
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc


matplotlib.use("Agg")
DEFAULT_WEIGHTS_FILE_NAME = 'doe_cnn.weights'
DEFAULT_MODEL_FILE_NAME = 'doe_cnn.model'
DEFAULT_LABEL_FILE_NAME = 'doe_cnn.labels.pickle'

CHANNELS = 5            # This will be redefined based on parameters
INIT_LR = 5e-1          # Default Loss Rate
# INIT_LR = 1e-2          # Default Loss Rate
INIT_DECAY = 1e-3       # Default Decay
# INIT_DECAY = 1e-1       # Default Decay
KERNEL_PIXELS = 17      # Default pixels by side on each tile
NUM_CLASSES = 2         # Default number of classes ("Geothermal", "Non-Geothermal")
BS = 32
EPOCHS = 500

class2code = {'none': 0,
              'Non-geothemal':1,
              'Geothemal':2}

code2class = {0: 'none',
              1: 'Non-geothemal',
              2: 'Geothemal'}


try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

''' Class definitions '''

print('Set-up complete.')

''' Main program '''

if __name__ == '__main__':
    ''' Main instructions '''
    print('Parsing input...')
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--channels", required=False, help='Number of channels in each image',
                    default=CHANNELS, type=int)
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-n", "--num_classes", required=False, help='Number of classes',
                    default=NUM_CLASSES, type=int)
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    dataset_path = args["dataset"]
    image_channels = args["channels"]
    num_classes = args["num_classes"]
    plot_file = args["plot"]
    true_random = args["true_random"]
    IMAGE_DIMS = (kernel_pixels, kernel_pixels, image_channels)
    BATCH_DIMS = (None, kernel_pixels, kernel_pixels, image_channels)
    # initialize the data and labels
    data = []
    labels = []
    ## grab the image paths and randomly shuffle them
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_files(dataset_path)))
    print('Number of images:', len(imagePaths))
    # Ensure 'random' numbers are not too random to compare networks
    if (not true_random):
        random.seed(42)
    random.shuffle(imagePaths)
    # leave just a subset of all images
    # imagePaths = imagePaths[:1400]
    ## loop over the input images
    img_count = 0
    for imagePath in imagePaths:
        # Reads image file from dataset
        image = np.load(imagePath)
        # Our Model uses (width, height, depth )
        data.append(image)
        # Gets label from subdirectory name and stores it
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    print('Read images:', len(data))
    # print('Read labels:', (labels))
    # scale the raw pixel intensities to the range [0, 1]
    data = np.asarray(data, dtype=np.float)
    data = np.nan_to_num(data)
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1024.0)))

    # binarize the labels
    lb = LabelBinarizer()
    labels_lb = lb.fit_transform(labels)

    y_binary = to_categorical(labels, num_classes=num_classes)
    y_inverse = argmax(y_binary, axis = 1)
    print('Read labels:', (lb.classes_))
    print('Transformed labels:', (y_binary[:10]))
    print('Inverted labels:', (y_inverse[:10]))
    print('Any nulls?: ', np.isnan(data).any())
    # sys.exit(0)  # exit after tests


    # partition the data into training and testing splits using 50% of
    # the data for training and the remaining 50% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
        y_binary, test_size=0.8, random_state=42)
    (validateX, testX, validateY, testY) = train_test_split(testX, testY,
            test_size=0.75, random_state=42)
    params = {'dim':(kernel_pixels,kernel_pixels), 'batch_size': batch_size,
            'n_classes': num_classes,
            'n_channels': image_channels,
            'shuffle': True, 'augment_data': augment_data}
    # construct the image generator for data augmentation
    my_batch_gen = DataGenerator(trainX, trainY, **params)
    print('creating generator with trainX, trainY of shapes: (%s, %s)' %
            (trainX.shape, trainY.shape)
            )
    ## initialize the model
    print("[INFO] compiling model...")
    print('SmallInception: (depth, width, height, classes) = (%s, %s, %s, %s)' %
           (IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], len(lb.classes_))
           )
    if (validate_only):
        print("[INFO] Skipping training...")
        print("[INFO] Validate-only model:", validate_only)
        pass
    else:
        print("[INFO] Training...")
        print("[INFO] Reset model:", reset_model)
        print("[INFO] Validate-only model:", validate_only)
        # opt=Adam(lr=INIT_LR, decay=INIT_DECAY)   # Old decay was: INIT_LR / EPOCHS)
        # opt = Adam(lr=INIT_LR, beta_1=INIT_DECAY, amsgrad=True)
        # opt = Adadelta(learning_rate=INIT_LR)
        opt = Adadelta()
        model3.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # define the network's early stopping
        print("[INFO] define early stop and auto save for network...")
        auto_save = ModelCheckpoint(model_file, monitor = 'val_accuracy', verbose = 0,
                                    save_best_only = True, save_weights_only=False,
                                    mode='auto')
        # can use validation set loss or accuracy to stop early
        # early_stop = EarlyStopping( monitor = 'val_accuracy', mode='max', baseline=0.97)
        early_stop = EarlyStopping( monitor = 'val_loss', mode='min', verbose=1, patience=50 )
        # train the network
        print("[INFO] training network...")
        # Train the model
        H = model3.fit(
            my_batch_gen,
            validation_data=(validateX, validateY),
            steps_per_epoch=len(trainX) // batch_size,
            # callbacks=[early_stop, auto_save],
            callbacks=[auto_save, early_stop],
            epochs=num_epochs, verbose=1)
        '''
        H = model3.fit_generator(
            generator = my_batch_gen,
            validation_data=(validateX, validateY),
            steps_per_epoch=len(trainX) // batch_size,
            # callbacks=[early_stop, auto_save],
            callbacks=[auto_save],
            epochs=num_epochs, verbose=1)
        '''
        # save the model to disk
        print("[INFO] serializing network...")
        model3.save( model_file )
        # save the label binarizer to disk
        print("[INFO] serializing label binarizer...")
        f = open(label_file, "wb")
        f.write(pickle.dumps(lb))
        f.close()

    # testY = lb.inverse_transform( testY ).astype(np.int64)
    testY = argmax(testY, axis=1)
    print('[INFO] Predicting ...')
    pre_y2 = model3.predict(testX, verbose = 1)
    print("pred set:", pre_y2[:10])
    pre_y2_prob = pre_y2
    pre_y2 = pre_y2.argmax(axis=-1)
    acc2 = accuracy_score(testY, pre_y2)
    print("test set:", testY[:10])
    print("pred set:", pre_y2[:10])
    print('Accuracy on test set: {0:.3f}'.format(acc2))
    print("Confusion Matrix:")
    print(confusion_matrix(testY, pre_y2))
    print()
    print("Classification Report")
    print(classification_report(testY, pre_y2))
    # Calculate ROC Curves if required
    if output_curves_file is not None:
        ROC_curve_calc( testY, pre_y2, class_num = 8, output_file_header = output_curves_file)
    if (not validate_only):
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = num_epochs
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        #plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"],
                label="val_accuracy")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig(plot_file)

