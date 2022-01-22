"""
  DOE Project : 
  Task        : 
  File        : 
  
    This is the main program to create, train and use an ANN to classify
  regions based on geothermal potential.


  @authors: Jim Moraga <jmoraga@mines.edu>
  @authors: Ebubekir Demir <edemir@mines.edu>
"""


# Ensure Python 2 & 3 compatibility
from __future__ import print_function
# import the necessary packages
import argparse
from imutils import paths
# from tensorflow import keras # not needed anymore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import AveragePooling2D, Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Nadam, Adadelta, Adagrad, Adamax, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import multi_gpu_model
# Progress bar
from tqdm import tqdm
# Matlab plotting library for charts
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
# Setup matplotlib
matplotlib.use("Agg")
# Create defaults for file names
DEFAULT_WEIGHTS_FILE_NAME = 'doe_cnn.weights'
DEFAULT_MODEL_FILE_NAME = 'doe_cnn.model'
DEFAULT_LABEL_FILE_NAME = 'doe_cnn.labels.pickle'
# Define defaults for input file parameters
CHANNELS = 5            # This will be redefined based on parameters
KERNEL_PIXELS = 17      # Default pixels by side on each tile
NUM_CLASSES = 2         # Default number of classes ("Geothermal", "Non-Geothermal")
# Default hyperparameters
INIT_LR = 5e-1          # Default Loss Rate
INIT_DECAY = 1e-3       # Default Decay
BS = 32
EPOCHS = 500
# Default labels
class2code = {'none': 0,
              'Non-geothemal':1,
              'Geothemal':2}
code2class = {0: 'none',
              1: 'Non-geothemal',
              2: 'Geothemal'}

# Ensure Python 3 is being called
try:
    raw_input          # Python 2
    print("Run in Python 3")
    raise SystemExit
except NameError:
    raw_input = input  # Python 3

''' Class definitions '''
class doe_ann_object(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

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
    print('Parsing input...')
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch_size", required=False,
                    help="Defines batch size", default = BS, type=int)
    ap.add_argument("-c", "--channels", required=False, help='Number of channels in each image',
                    default=CHANNELS, type=int)
    ap.add_argument("-e", "--epochs", required=False, type=int,
                    help="Number of epochs to train)", default=EPOCHS)
    ap.add_argument("-g", "--gpus", required=False, type=int,
                    help="Number of GPUs to run in parallel", default=1)
    ap.add_argument("-i", "--image", required=True,
                    help="path to input multi-band image (i.e., image file name)")
    ap.add_argument("-k", "--kernel_size", required=False,
                    help='Number of pixels by side in each image',
                    default=KERNEL_PIXELS, type=int)
    ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    ap.add_argument("-n", "--num_classes", required=False, help='Number of classes',
                    default=NUM_CLASSES, type=int)
    ap.add_argument("-p", "--prediction_map", required=False, type=str, default="prediction_map.npy",
                    help="path to output prediction map (ending in .npy)")
    ap.add_argument("-o", "--output_raster", required=False, type=str, help="Raster file ending in .gri",
                    default = "prediction_raster.gri")
    args = vars(ap.parse_args())
    batch_size = args["batch_size"]
    num_channels = args["channels"]
    num_epochs = args["epochs"]
    num_gpus = args["gpus"]
    image_name = args["image"]
    kernel_size = args["kernel_size"]
    label_file = args["labelbin"]
    model_file = args["model"]
    num_classes = args["num_classes"]
    prediction_map = args["prediction_map"]
    output_raster = args["output_raster"]
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
    # Read model file
    print('[INFO] Loading model from file...')
    model3 = load_model( model_file )
    model3.summary()
    # Get rid of NaN's
    img_b = np.array(img_b, dtype=np.float64)
    img_b = np.nan_to_num(img_b)
    print("Image shape:", img_b.shape)
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
    ### Check whether multi-gpu option was enabled
    print('[INFO] Creating prediction map...')
    for i in tqdm(range(img_x), desc="Predicting...",
                  ascii=False, ncols=75):
        # initialize the data and labels
        data = []
        labels = []
        ## loop over the input images
        img_count = 0
        for j in range(img_y):
            image = sc.apply_mask(i+PADDING, j+PADDING)
            data.append(image)
        data = np.array(data, dtype=np.float64)
        data = np.nan_to_num(data)
        pre_y = model3.predict( data, verbose = 0 )
        pre_y = pre_y.argmax(axis=-1)
        new_map[i,:] = pre_y
    new_map = np.asarray(new_map)
    print('saving file:', prediction_map)
    f = open(prediction_map, 'wb')
    np.save(f, new_map)
    inDs = gdal.Open(image_name)
    band1 = inDs.GetRasterBand(1)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    cropData = band1.ReadAsArray(0,0,cols,rows)
    driver = inDs.GetDriver()
    outDs = driver.Create(output_raster, cols, rows, 1, GDT_Int32)
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
    print("saving file: ", output_raster)
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
