"""
  DOE Project : 
  Task        : 
  File        : 
  
    This is the main program to create, train and use an ANN to classify
  regions based on geothermal potential.


  @authors: Jim Moraga <jmoraga@mines.edu>
"""


from __future__ import print_function
import os
# Source: https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# https://www.tensorflow.org/api_docs/python/tf/autograph/set_verbosity
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
import tensorflow as tf
tf.get_logger().setLevel("WARNING")

# import the necessary packages
import argparse
# import keras
from tqdm import tqdm
import jigsaw
from jigsaw import DataGenerator
from jigsaw.rasterspcv import RasterSpCV

if tf.__version__ >= "2.4.0":
    from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
else:
    from tensorflow.keras.utils import multi_gpu_model
try:
    from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
except:
    pass
try:
    from tensorflow.keras.utils import multi_gpu_model
except:
    pass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import os
import os.path as path
import sys
import pickle
import random
import rasterio
import skimage
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta #, Adam, Nadam, Adagrad, Adamax, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model



matplotlib.use("Agg")
DEFAULT_WEIGHTS_FILE_NAME = 'doe_cnn.weights'
DEFAULT_MODEL_FILE_NAME = 'doe_cnn.model'
DEFAULT_LABEL_FILE_NAME = 'doe_cnn.labels.pickle'

INIT_LR = 5e-1          # Default Loss Rate
INIT_DECAY = 1e-3       # Default Decay
CHANNELS = 5            # This will be redefined based on parameters
KERNEL_PIXELS = 17      # Default pixels by side on each tile
BS = 32
EPOCHS = 500
NUM_CLASSES = 2

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


class doe_ann_object(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir


''' Function definitions '''
def ROC_curve_calc( testY, pre_y2, class_num, output_file_header ):
    '''
    Calculate other stats
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes=class_num
    for i in range(1, n_classes):
        class_plot = i
        true_labels = (testY[:]==i)
        pred_probs = (pre_y2[:]==i)
        if class_plot == 0:
            pred_probs = pred_probs*0
            pred_probs = pre_y2_prob[:,0]*0
        else:
            pred_probs = pre_y2_prob[:,class_plot-1]
        print('For class ', i, ': shape = ', true_labels.shape)
        print('For true_labels, first item = ', true_labels[0])
        fpr[i], tpr[i], _ = roc_curve( true_labels, pred_probs) # , pos_label = 1)
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(1, n_classes):
        # Plot for a class
        class_plot = i
        class_name = code2class[class_plot]
        plt.figure()
        line_width = 2
        plt.plot( fpr[class_plot], tpr[class_plot], color='darkorange',
                  lw = line_width, label = 'ROC Curve for class %s (area = %0.2f)'%( class_name, roc_auc[class_plot] ))
        plt.plot([0,1], [0, 1], color='navy', lw=line_width, linestyle='--')
        plt.xlim( [0.0, 1.0] )
        plt.ylim( [0.0, 1.05] )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig(output_file_header+'.'+class_name+'.png')
    return

print('Set-up complete.')

''' Main program '''
if __name__ == '__main__':
    ''' Main instructions '''
    print('Parsing input arguments...')
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augment", required=False,
                    help="Augment images by rotating and flipping horizontally/vertically",
                    dest='augment', action = 'store_true', default = False)
    ap.add_argument("-b", "--batch_size", required=False,
                    help="Defines batch size", default = BS, type=int)
    ap.add_argument("-c", "--channels", required=False, help='Number of channels in each image',
                    default=CHANNELS, type=int)
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset (i.e., image file name)")
    ap.add_argument("-e", "--epochs", required=False, type=int,
                    help="Number of epochs to train)", default=EPOCHS)
    ap.add_argument("-g", "--gpus", required=False, type=int,
                    help="Number of GPUs to run in parallel", default=1)
    ap.add_argument("-i", "--internal_size", required=False, type=int,
                    help="Maximum size of interal kernels", default=9)
    ap.add_argument("-k", "--kernel_pixels", required=False,
                    help='Number of pixels by side in each image',
                    default=KERNEL_PIXELS, type=int)
    ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    ap.add_argument("-n", "--num_classes", required=False, help='Number of classes',
                    default=NUM_CLASSES, type=int)
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    ap.add_argument("-r", "--reset", required=False,
                    help="Don't load setup files, train from scratch",
                    dest='reset', action = 'store_true', default = False)
    ap.add_argument("-o", "--output_curves", required=False, help="Starting file name for ROC curves",
                    default = None)
    ap.add_argument("-s", "--num_samples", required=False, help='Number of samples to use. It correspond to number of each class if balanced, or total if unbalanced (see --unbalanced)',
                    default=100, type=int)
    ap.add_argument("-t", "--true_random", required=False,
                    help="Ensure true random shuffling of training and test sets",
                    dest='true_random', action = 'store_true', default = False)
    ap.add_argument("-v", "--validate", required=False,
                    help="Don't train, only validate with random images from dataset",
                    dest='validate', action = 'store_true', default = False)
    ap.add_argument("-w", "--weights", required=False, help="path to input or output model weights",
                    default = None)
    ap.add_argument("-z", "--unbalanced", required=False,
                    help="Don't balance images for each class in the dataset used for training",
                    dest='unbalanced', action = 'store_true', default = False)
    args = vars(ap.parse_args())
    print(args)
    augment_data = args["augment"]
    batch_size = args["batch_size"]
    image_channels = args["channels"]
    dataset_path = args["dataset"]
    num_epochs = args["epochs"]
    num_gpus = args["gpus"]
    internal_size = args["internal_size"]
    kernel_pixels = args["kernel_pixels"]
    label_file = args["labelbin"]
    model_file = args["model"]
    num_classes = args["num_classes"]
    plot_file = args["plot"]
    reset_model = args["reset"]
    num_samples = args["num_samples"]
    true_random = args["true_random"]
    validate_only = args["validate"]
    weights_file = args["weights"]
    balanced_images = not (args["unbalanced"])
    output_curves_file = args["output_curves"]
    if reset_model:
        print("[INFO] Reset model")
        model_exist = False
        weights_exist = False
    else:
        print("[INFO] Don't reset model")
        # Ensures model file exists and is really a file
        if model_file is not None:
            try:
                assert path.exists(model_file), 'weights file {} does not exist'.format(model_file)
                assert path.isfile(model_file), 'weights path {} is not a file'.format(model_file)
                model_exist = True
            except:
                model_exist = False
        else:
            model_file = DEFAULT_MODEL_FILE_NAME
            model_exist = False
        # Ensures weights file exists and is really a file
        if weights_file is not None:
            assert path.exists(weights_file), 'weights file {} does not exist'.format(weights_file)
            assert path.isfile(weights_file), 'weights path {} is not a file'.format(weights_file)
            weights_exist = True
        else:
            weights_file = DEFAULT_WEIGHTS_FILE_NAME
            weights_exist = False
    IMAGE_DIMS = (kernel_pixels, kernel_pixels, image_channels)
    BATCH_DIMS = (None, kernel_pixels, kernel_pixels, image_channels)
    # initialize the data and labels
    data = []
    labels = []

    ### Load data
    ## grab the image paths and randomly shuffle them
    print("[INFO] loading images...")
    # Ensure 'random' numbers are not too random to compare networks
    if (not true_random):
        random.seed(42)
        random_state=42
    else:
        random_state=None
    rSpCV = RasterSpCV(dataset_path, 
                       num_channels=image_channels,
                       kernel_size=kernel_pixels,
                       sample=num_samples,
                       balanced=balanced_images,
                       random_state=random_state,
                       verbose=2)
    cv = rSpCV.SpatialCV()
    print('Number of images:', len(rSpCV))
    # scale the raw pixel intensities to the range [0, 1]
    # data = rSpCV.X()
    print("Loading images...")
    # data = data[np.arange(len(data))]
    data = []
    #### for imagePath in imagePaths:
    for i in tqdm(range(len(rSpCV)), desc="Loading...",
                          ascii=False, ncols=75):
        # Reads image file from dataset
        # Our Model uses (width, height, depth )
        data.append(rSpCV[i])
    print('Read images:', i+1)
    # print('Read images:', len(data))
    # scale the raw pixel intensities to the range [0, 1]
    data = np.asarray(data, dtype=np.float64)

    labels = rSpCV.y()
    print("Loaded:", type(data), ". shape: ", data.shape)
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
    ### END Reading images
    #
    ###
    # partition the data into training and testing splits using 20% of
    # the data for training and the remaining 80% for testing/validation
    (trainX, testX, trainY, testY) = train_test_split(data,
        y_binary, test_size=0.5, random_state=42) # Was 0.8
    (validateX, testX, validateY, testY) = train_test_split(testX, testY,
            test_size=0.75, random_state=42) # was 0.75
    params = {'dim':(kernel_pixels,kernel_pixels), 'batch_size': batch_size,
            'n_classes': num_classes,
            'n_channels': image_channels,
            'shuffle': True, 'augment_data': augment_data}
    # construct the image generator for data augmentation
    my_batch_gen = DataGenerator(trainX, trainY, **params)
    print('creating generator with trainX, trainY of shapes: (%s, %s)' %
            (trainX.shape, trainY.shape)
            )

    ###
    '''
    Creates the network
    '''
    if (reset_model or not model_exist):
        print('[INFO] Building model from scratch...')
        # Builds model
        model3 = jigsaw.build_jigsaw( internal_size = internal_size, # Max internal kernel 9x9 or 13x13
                                      num_classes = num_classes,
                                      image_dim = IMAGE_DIMS )
        model3.summary()
    else:
        # Builds model
        print('[INFO] Loading model from file...')
        model3 = load_model( model_file )
        model3.summary()
    ### Check whether multi-gpu option was enabled
    ### Careful, no hardware validation on number of GPUs
    if (num_gpus>1):
        try:
            model3 = multi_gpu_model( model3, gpus = num_gpus )
        except:
            pass
    ## initialize the model
    print("[INFO] compiling model...")
    print('SmallJigsaw: (depth, width, height, classes) = (%s, %s, %s, %s)' %
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
        # Hyperparameter selection: Optimization function
        # opt = Adam(lr=INIT_LR, decay=INIT_DECAY)   # Old decay was: INIT_LR / EPOCHS)
        # opt = Adam(lr=INIT_LR, beta_1=INIT_DECAY, amsgrad=True)
        # opt = Adadelta(learning_rate=INIT_LR)
        opt = Adadelta()
        model3.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # define the network's early stopping
        print("[INFO] define early stop and auto save for network...")
        auto_save = ModelCheckpoint(model_file, monitor = 'accuracy', verbose = 0, # vas 'val_accuracy'
                                    save_best_only = True, save_weights_only=False,
                                    mode='auto', save_freq=200)
        # can use validation set loss or accuracy to stop early
        # early_stop = EarlyStopping( monitor = 'val_accuracy', mode='max', baseline=0.97)
        # patience was 50
        early_stop = EarlyStopping( monitor = 'val_loss',
                                min_delta=0.001,
                                mode='auto',
                                verbose=1, patience=5)
        early_stop2 = EarlyStopping( monitor = 'val_accuracy',
                                min_delta=0.001,
                                mode='auto',
                                verbose=1, patience=5)
        # train the network
        print("[INFO] training network...")
        # Train the model
        H = model3.fit(
            my_batch_gen,
            validation_data=(validateX, validateY),
            steps_per_epoch=len(trainX) // batch_size,
            callbacks=[auto_save, early_stop],
            epochs=num_epochs, verbose=1)
        # save the model to disk
        print("[INFO] serializing network...")
        model3.save( model_file )
        # save the label binarizer to disk
        print("[INFO] serializing label binarizer...")
        f = open(label_file, "wb")
        f.write(pickle.dumps(lb))
        f.close()
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
    print(confusion_matrix(y_true = testY, y_pred = pre_y2))
    print(confusion_matrix(y_true = testY, y_pred = pre_y2, normalize='all'))
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
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"],
                label="val_accuracy")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig(plot_file)
