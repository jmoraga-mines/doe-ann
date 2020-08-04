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
import doe_tiff as dt
from osgeo import gdal
from osgeo.gdalconst import *



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




'''
# In case we need to append directories to Python's path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
#sys.path.append(os.path.join(ROOT_DIR, 'a_directory'))
'''

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

''' Class definitions '''
class doe_ann_object(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_set, labels, batch_size=BS,
            dim=(KERNEL_PIXELS,KERNEL_PIXELS,CHANNELS),
            n_channels=CHANNELS, n_classes=NUM_CLASSES,
            shuffle=True, augment_data=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_set = data_set
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment_data = augment_data
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_set) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_set))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
#        if (self.augment_data):
#            print('Generating augmented data...')
#        else:
#            print('Generating data...')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image_raw = self.data_set[ID]
            if self.augment_data:
                if np.random.randint(1):
                    image_raw = np.rot90( image_raw, 2 )
                if np.random.randint(1):
                    image_raw = np.fliplr( image_raw )
                if np.random.randint(1):
                    image_raw = np.flipud( image_raw )
            X[i,] = image_raw
            # Store class
            y[i] = self.labels[ID]
        return X, y


def inception_m( input_net, first_layer = None ):
    conv1 = Conv2D(128, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(input_net)
    inception_t1_1x1 = Conv2D(256, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_3x3_reduce = Conv2D(96, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_3x3 = Conv2D(128, (3,3), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_3x3")(inception_t1_3x3_reduce)
    inception_t1_5x5_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_5x5 = Conv2D(128, (5,5), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_5x5")(inception_t1_5x5_reduce)
    inception_t1_7x7_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_7x7 = Conv2D(128, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_7x7")(inception_t1_7x7_reduce)
    inception_t1_9x9_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_9x9 = Conv2D(64, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_9x9")(inception_t1_9x9_reduce)
    inception_t1_11x11_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_11x11 = Conv2D(64, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_11x11")(inception_t1_11x11_reduce)
    inception_t1_13x13_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_13x13 = Conv2D(64, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_13x13")(inception_t1_13x13_reduce)
    inception_t1_15x15_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_15x15 = Conv2D(64, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_15x15")(inception_t1_15x15_reduce)
    inception_t1_17x17_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_17x17 = Conv2D(64, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_17x17")(inception_t1_17x17_reduce)
    inception_t1_19x19_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    inception_t1_19x19 = Conv2D(64, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_19x19")(inception_t1_19x19_reduce)
    inception_t1_pool = MaxPooling2D(pool_size=(3,3), strides = (1,1), padding='same')(conv1)
    inception_t1_pool_proj = Conv2D(32, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(inception_t1_pool)
    if first_layer is None:
        inception_t1_output = Concatenate(axis = -1)([inception_t1_1x1, inception_t1_3x3, inception_t1_5x5,
                                                      inception_t1_7x7, inception_t1_9x9, inception_t1_11x11, 
                                                      inception_t1_13x13, inception_t1_15x15, 
                                                      inception_t1_17x17, inception_t1_19x19, inception_t1_pool_proj])
    else:
        inception_t1_first = Conv2D(96, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(first_layer)
        inception_t1_output = Concatenate(axis = -1)([inception_t1_first, inception_t1_1x1, inception_t1_3x3,
                                                      inception_t1_5x5, inception_t1_7x7, inception_t1_9x9, inception_t1_11x11, 
                                                      inception_t1_13x13, inception_t1_15x15, 
                                                      inception_t1_17x17, inception_t1_19x19, inception_t1_pool_proj])
    return inception_t1_output

def inception_m_end( input_net, num_classes = NUM_CLASSES, first_layer = None ):
    avg_pooling = AveragePooling2D(pool_size=(3,3), strides=(1,1), name='avg_pooling')(input_net)
    flat = Flatten()(avg_pooling)
    flat = Dense(16, kernel_regularizer=l2(0.002))(flat)
    flat = Dropout(0.4)(flat)
    if first_layer is not None:
        input_pixel = Flatten()(first_layer)
        input_pixel = Dense(16, kernel_regularizer=l2(0.002))(input_pixel)
        input_pixel = Dropout(0.2)(input_pixel)
        input_pixel = Dense(16, kernel_regularizer=l2(0.002))(input_pixel)
        input_pixel = Dropout(0.2)(input_pixel)
        flat = Concatenate(axis = -1)([input_pixel, flat])
    flat = Dense(32, kernel_regularizer=l2(0.002))(flat)
    avg_pooling = Dropout(0.4)(flat)
    loss3_classifier = Dense(num_classes, kernel_regularizer=l2(0.002))(avg_pooling)
    loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)
    return loss3_classifier_act


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
        # class_indexes = np.where( testY == i )
        # print('For testY, shape = ', testY.shape)
        # print('For testY, first item = ', testY[0])
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
        # Compute micro-average ROC curve and area
        # fpr["micro"], tpr["micro"], _ = roc_curve( testY.ravel(), pre_y2.ravel() )
        # fpr["micro"], tpr["micro"], _ = roc_curve( testY, pre_y2 )
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
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
    print('Parsing input...')
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch_size", required=False,
                    help="Defines batch size", default = BS, type=int)
    ap.add_argument("-c", "--channels", required=False, help='Number of channels in each image',
                    default=CHANNELS, type=int)
    ap.add_argument("-e", "--epochs", required=False, type=int,
                    help="Number of epochs to train)", default=EPOCHS)
    ap.add_argument("-i", "--image", required=True,
                    help="path to input multi-band image (i.e., image file name)")
    ap.add_argument("-k", "--kernel_size", required=False,
                    help='Number of pixels by side in each image',
                    default=KERNEL_PIXELS, type=int)
    ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    ap.add_argument("-n", "--num_classes", required=False, help='Number of classes',
                    default=NUM_CLASSES, type=int)
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    ap.add_argument("-o", "--output_curves", required=False, help="Starting file name for ROC curves",
                    default = None)
    args = vars(ap.parse_args())
    batch_size = args["batch_size"]
    num_channels = args["channels"]
    num_epochs = args["epochs"]
    image_name = args["image"]
    kernel_size = args["kernel_size"]
    label_file = args["labelbin"]
    model_file = args["model"]
    num_classes = args["num_classes"]
    plot_file = args["plot"]
    output_curves_file = args["output_curves"]
    # Ensures model file exists and is really a file
    PADDING = int(kernel_size/2)
    try:
        assert path.exists(model_file), 'Model path {} does not exist'.format(model_file)
        assert path.isfile(model_file), 'Model file {} is not a file'.format(model_file)
        model_exist = True
    except:
        model_exist = False
        raise FileNotFoundError
    try:
        assert path.isfile(image_name), 'Image file {}: is not a file'.format(model_file)
        img_b = dt.io.read_gdal_file(image_name)
        max_channels = img_b.shape[2]
    except:
        print("Image file not found or erroneous")
        raise FileNotFoundError
    weights_exist = False

    # Get rid of NaN's
    img_b = np.array(img_b, dtype=np.float)
    img_b = np.nan_to_num(img_b)
    # By Jim: to reduce the size of the input tiff
    print("Image shape:", img_b.shape)
    # img_b = img_b[200:500,1000:,:]
    print("Resized image shape:", img_b.shape)
    assert num_channels>0, 'Channels has to be a positive integer'
    assert num_channels<=max_channels, 'Channels has to be equal or lower than {}'.format(max_channels)
    img_b_scaled = img_b[:, :, 1:num_channels+1]
    print("DLM input image shape:", img_b_scaled.shape)
    mask_b = img_b[:, :, 0] # first band is Ground Truth
    (img_x, img_y) = mask_b.shape
    print("Mask shape:", mask_b.shape)
    new_map = np.zeros_like(mask_b) # creates empty map
    print("Map shape:", new_map.shape)
    for i in range(0, img_b_scaled.shape[2]):
        print("band (",i,") min:", img_b_scaled[:,:,i].min())
        print("band (",i,") max:", img_b_scaled[:,:,i].max())
    img_b_scaled = dt.frame_image(img_b_scaled, PADDING)
    sc = dt.GeoTiffConvolution(img_b_scaled, kernel_size, kernel_size)

    IMAGE_DIMS = (kernel_size, kernel_size, num_channels)
    BATCH_DIMS = (None, kernel_size, kernel_size, num_channels)
    # Builds model
    print('[INFO] Loading model from file...')
    model3 = load_model( model_file )
    # model3.summary()
    print('[INFO] Creating prediction map...')
    for i in range(img_x):
        # initialize the data and labels
        data = []
        labels = []
        ## loop over the input images
        img_count = 0
        for j in range(img_y):
            image = sc.apply_mask(i+PADDING, j+PADDING)
            data.append(image)
            # label = mask_b[i, j]
            # labels.append(label)
        data = np.array(data, dtype=np.float)
        data = np.nan_to_num(data)
        # print("Cropped image shape:", image.shape)
        # print("first image shape:", data[0].shape)
        # print("number of images:", len(data))
        pre_y = model3.predict( data, verbose = 0 )
        pre_y = pre_y.argmax(axis=-1)
        new_map[i,:] = pre_y
    new_map = np.asarray(new_map)
    file_name = "prediction_map.npy"
    print('saving file:', file_name)
    f = open(file_name, 'wb')
    np.save(f, new_map)
    inDs = gdal.Open(image_name)
    band1 = inDs.GetRasterBand(1)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    cropData = band1.ReadAsArray(0,0,cols,rows)
    driver = inDs.GetDriver()
    outDs = driver.Create("prediction_raster.gri", cols, rows, 1, GDT_Int32)
    outBand = outDs.GetRasterBand(1)
    outData = new_map.T
    outBand.WriteArray(outData, 0, 0)
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())
    outDs.FlushCache()
    del outData
    outDs = None
    print("saving file: prediction_raster.gri")


    # Calculate accuracy
    new_map = new_map.flatten()
    mask_b = mask_b.flatten()
    acc2 = accuracy_score(mask_b, new_map)
    print('Accuracy on test set: {0:.3f}'.format(acc2))
    print("Confusion Matrix:")
    print(confusion_matrix(mask_b, new_map))
    print()
    print("Classification Report")
    print(classification_report(mask_b, new_map))

