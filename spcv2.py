import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Use this before loading tensorflow
import random
import numpy as np
import tensorflow as tf

def random_seed():
    return os.urandom(42)

def reset_seeds(random_state = 42):
    try:
        tf.keras.utils.set_random_seed(random_state)
        return 0
    except: 
        random.seed(random_state)
        np.random.seed(random_state)
        tf.random.set_seed(random_state) # Tensorflow 2.9
    try:
        from tensorflow import set_random_seed # Tensorflow 1.x
        set_random_seed(random_state)
        return 2
    except:
        pass
    return 1

max_gpus = len(tf.config.list_physical_devices('GPU'))
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
"""
Reset all random seeds
"""               
reset_seeds(123)
#### Basic set-up
tif_file_name ='brady_ai_stack.grd'
image_channels = 7 # Was 9
#tif_file_name ='/store03/thesis/git/doe-som/brady_som_output.grd'
#image_channels = 3

num_samples = 2000
kernel_pixels = 19
my_patience = 5
kernel_internal = 13

print('[Info] *** Configuration ***')
print('File     : ', tif_file_name)
print('Bands    : ', image_channels)
print('Samples  : ', num_samples)
print('Kernel   : ', kernel_pixels, ' pixels per side')
print('Channels : ', image_channels)
print('CNN size : (', kernel_internal, 'x', kernel_internal, ') maximum kernel size')
print('Patience : ', my_patience)
print('GPUs max : ', max_gpus)
num_epochs = 150
batch_size = 200

BS=batch_size

NUM_CLASSES = 2         # Default number of classes ("Geothermal", "Non-Geothermal")
IMAGE_DIMS = (kernel_pixels, kernel_pixels, image_channels)
BATCH_DIMS = (None, kernel_pixels, kernel_pixels, image_channels)
KERNEL_PIXELS = kernel_pixels
CHANNELS = image_channels

num_classes = NUM_CLASSES
augment_data = True

# Load libraries
# import gc
# from matplotlib import pyplot

print('[Info] *** Configuration ***')
print('Classes    : ', num_classes)
print('Epochs     : ', num_epochs)
print('Batch size : ', batch_size)

# Helper class, generates data for keras
from jigsaw.jigsaw import DataGenerator
from jigsaw.rasterspcv import Kernel3D, GeoTiffSlicer
from jigsaw.rasterspcv import RasterSpCV
from jigsaw import build_jigsaw

# Builds model

rSpCV = RasterSpCV(tif_file_name, kernel_pixels, num_channels = image_channels, sample = 100000, verbose=0, augment = True)

#### Here starts the good part

reset_seeds()
cv = rSpCV.SpatialCV()
X = rSpCV.X()
y = rSpCV.y()
y_binary = to_categorical(y)

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score, f1_score
from keras.wrappers.scikit_learn import KerasClassifier
def BAcc(y_true, y_pred):
    #y_pred = np.ravel(y_true[:,1])
    y_true = np.ravel(y_true[:,1])
    #y_pred = np.argmax(y_pred, axis=-1)
    assert(len(y_true)==len(y_pred))
    b_acc = balanced_accuracy_score(y_true, y_pred)
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred)
    print("Acc  :", acc)
    print("Bacc :", b_acc)
    print("F-1  :", f1)
    return b_acc
    print("Acc:", np.count_nonzero(y_true==y_pred)/len(y_true))
    #print('y_true:', y_true)
    #print('y_pred:', y_pred)
    l = []
    for i in np.unique(y_true):
        a = np.ravel(np.asarray(y_true==(i)).nonzero())
        #print("Index:", a)
        a_score = np.count_nonzero(y_true[a]==y_pred[a])/len(a)
        l.append(a_score)
    b_acc = np.mean(l)
    #print("L   :",l)
    print("Bacc: ", b_acc)
    return b_acc

balanced_accuracy = make_scorer(BAcc)

m = KerasClassifier(build_jigsaw, internal_size = kernel_internal,
                    num_classes = num_classes,
                    image_dim = IMAGE_DIMS,
                    batch_size = 10, epochs = 10)

### Now run cross-validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(m, X, y_binary, cv=cv, scoring = balanced_accuracy)

