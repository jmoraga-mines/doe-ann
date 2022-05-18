import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#tif_file_name ='brady_ai_stack.grd'
#image_channels = 7 # Was 9
tif_file_name ='../doe-som/brady_som_output.grd'
image_channels = 3
print('File  : ', tif_file_name)
print('Bands : ', image_channels)

from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Nadam, Adadelta, Adagrad, Adamax, SGD
from tensorflow.keras.utils import Sequence
import numpy as np

from tensorflow.keras.utils import to_categorical

num_epochs = 150
num_samples = 200000
print('Samples : ', num_samples)
BS=200
kernel_pixels = 19
batch_size = BS

NUM_CLASSES = 2         # Default number of classes ("Geothermal", "Non-Geothermal")
IMAGE_DIMS = (kernel_pixels, kernel_pixels, image_channels)
BATCH_DIMS = (None, kernel_pixels, kernel_pixels, image_channels)
KERNEL_PIXELS = kernel_pixels
CHANNELS = image_channels

num_classes = NUM_CLASSES
augment_data = True


class DataGenerator(Sequence):
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


class Kernel3D:
    def __init__(self, rows=3, cols=3, shape='rect', radius=None, no_value=np.NaN):
        if shape == 'circle':
            self.rows = 2*radius+1
            self.cols = 2*radius+1
            self.mask = self.round_mask(radius)
            self.row_buffer = radius
            self.col_buffer = radius
        else:
            self.rows = rows
            self.cols = cols
            self.mask = np.ones((rows, cols))
            self.row_buffer = int((rows-1)/2)
            self.col_buffer = int((cols-1)/2)
        self.mask = self.mask[np.newaxis, :, :]
        self.no_value = no_value
        assert((rows%2) == 1)
        assert((cols%2) == 1)

    def round_mask(self, radius):
        diameter = 2*radius+1
        mask = np.empty((diameter, diameter,))
        mask[:] = self.no_value
        sq_radius = radius**2
        for i in range(diameter):
            for j in range(diameter):
                if ((i-radius)**2+(j-radius)**2) <= sq_radius:
                    mask[i, j] = 1
        return mask

    def getSubset(self, matrix, row, column):
        m_rows = matrix.shape[1]
        assert (row >= self.row_buffer), f"Out of bounds row {row}, from {m_rows}"
        assert (row < (m_rows-self.row_buffer)), f"Out of bounds row {row}, from {m_rows}"
        m_cols = matrix.shape[2]
        assert((column >= self.col_buffer) and (column < (m_cols-self.col_buffer))), f"Out of bounds column {column}, from {m_cols}"
        row_start = row-self.row_buffer
        row_end = row+self.row_buffer
        column_start = column-self.col_buffer
        column_end = column+self.col_buffer
        small_matrix = matrix[:, row_start:row_end+1, column_start:column_end+1]
        return small_matrix*self.mask

    def getPercentage(self, matrix, row, column):
        test_matrix = self.getSubset(matrix, column, row)
        return test_matrix.mean()

class GeoTiffSlicer(object):
    def __init__(self, land_matrix, kernel_rows=None, kernel_cols=None,
                 kernel_shape='rect', kernel_radius=0, no_value = np.NaN):
        # (d, h, w) input tiff from rasterio
        if kernel_cols is None:
            kernel_cols = kernel_rows
        assert(kernel_cols < land_matrix.shape[2])
        assert(kernel_rows < land_matrix.shape[1])
        assert((kernel_shape == 'rect') or (kernel_shape == 'circle'))
        assert(kernel_radius>=0)
        if kernel_shape == 'rect':
            self.kernel = Kernel3D(rows=kernel_rows, cols=kernel_cols)
        else:
            self.kernel = Kernel3D(radius=kernel_radius,
                                   shape=kernel_shape,
                                   no_value=no_value)
            kernel_rows = kernel_cols = 2*kernel_radius+1
        self.kernel_rows = kernel_rows
        self.kernel_cols = kernel_cols
        self.land_matrix = land_matrix
        self.land_matrix_channels, self.land_matrix_cols, self.land_matrix_rows = land_matrix.shape
        self.land_matrix_cols = land_matrix.shape[2]
        self.land_matrix_rows = land_matrix.shape[1]
        self.land_matrix_channels = land_matrix.shape[0]
        self.small_row_min = self.kernel.row_buffer
        self.small_row_max = self.land_matrix_rows - self.small_row_min
        self.small_column_min = self.kernel.col_buffer
        self.small_column_max = self.land_matrix_cols - self.small_column_min

    def apply_mask(self, row, column):
        return self.kernel.getSubset(self.land_matrix, row=row, column=column)

    def calculate(self):
        m1 = np.zeros_like(self.land_matrix, dtype='float')
        for j in range(self.small_row_min, self.small_row_max):
            for i in range(self.small_column_min, self.small_column_max):
                m1[i, j] = self.kernel.getPercentage(self.land_matrix, column=i, row=j)
        return m1

from pandas.core.dtypes.missing import isna
import rasterio
from sklearn.model_selection import GroupKFold
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.model_selection import BaseCrossValidator
from abc import  ABC, abstractmethod, ABCMeta
import pandas as pd
import geopandas as gpd

class RasterSpCV(BaseCrossValidator, list, ABC):
  def __init__(self, base_image, kernel_size,
               no_value = np.NaN, verbose = 0,
               partitions = 5,
               sample = None, random_state = None):
    assert isinstance(base_image, rasterio.io.DatasetReader), f"Wrong type, received {type(base_image)}"
    try:
      c, h, w = base_image.count, base_image.height, base_image.width
    except:
      c=h=w=0
    assert (c>1) and (h>kernel_size*2) and  (w>kernel_size*2)
    assert ((isinstance(sample, type(None))) or (isinstance(sample, int))), f"Wrong type, received {type(sample)}"
    assert ((kernel_size>0) and ((kernel_size%2)==1))
    self.kernel_size = kernel_size
    self.verbose = verbose
    xmin, ymax = np.around(base_image.xy(0.00, 0.00), 8)  # millimeter accuracy for longitude
    xmax, ymin = np.around(base_image.xy(h-1, w-1), 8)  # millimeter accuracy
    tif_x = np.linspace(xmin, xmax, w)
    tif_y = np.linspace(ymax, ymin, h) # coordinates are top to bottom
    tif_col = np.arange(w)
    tif_row = np.arange(h)#[::-1] # This will match numpy array location
    xs, ys = np.meshgrid(tif_x, tif_y)
    cs, rs = np.meshgrid(tif_col, tif_row)
    zs = base_image.read(1) # First band contains categories
    if(verbose>0):
      zs_u = len(np.unique(zs))
      if(zs_u<2):
        print("Warning, ", zs_u, " output categories is less than 2")
    tif_mask = base_image.read_masks(1) > 0
    # Just keep valid points (non-NaN)
    xs, ys = xs[tif_mask], ys[tif_mask]
    cs, rs, zs = cs[tif_mask], rs[tif_mask], zs[tif_mask]
    data = {'Column': pd.Series(cs.ravel()),
            'Row': pd.Series(rs.ravel()),
            'x': pd.Series(xs.ravel()),
            'y': pd.Series(ys.ravel()),
            'z': pd.Series(zs.ravel())}
    df = pd.DataFrame(data=data)
    df = df.dropna()
    geometry = gpd.points_from_xy(df.x, df.y)
    tif_crs = base_image.crs
    gdf = gpd.GeoDataFrame(df, crs=tif_crs, geometry=geometry) # [['z', 'Column', 'Row']]
    if(not isinstance(sample, type(None))):
      gdf=gdf.sample(n = sample, random_state=random_state)
    if(verbose>0):
      print("Splitting input layer data with MiniBatchKMeans")
    km_spcv = MiniBatchKMeans(n_clusters = partitions,
                              random_state=random_state)
    tif_folding = gdf.copy()
    tif_folding['Fold'] = -1
    km_spcv_model = km_spcv.fit(tif_folding[['x', 'y']])
    labels = km_spcv_model.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    if(verbose>0):
      print("Counts:", counts)
      print("Labels: #", len(labels))
      print("Partitions:", unique_labels)
    assert (len(unique_labels)==partitions)
    tif_folding['Fold'] = (labels)
    self.folding = tif_folding
    n_splits = len(unique_labels)
    self.partitions = unique_labels
    _land_matrix = base_image.read()
    _land_matrix = _land_matrix[1:(CHANNELS+1), :, :]
    _land_matrix = np.nan_to_num(_land_matrix)
    _land_matrix = np.pad(_land_matrix,
                          pad_width=((0,0),
                                     (kernel_size, kernel_size),
                                     (kernel_size, kernel_size)),
                          mode='symmetric')
    if(verbose>0):
      print("Shape of new tiff:", _land_matrix.shape)
    self.slicer = GeoTiffSlicer(land_matrix=_land_matrix,
                                kernel_rows=kernel_size)
    self.shape = (len(self.folding.index), kernel_size, kernel_size, c-1)

  def SpatialCV_split(self):
    for fold_index in self.partitions:
      if(self.verbose>0):
        print("Fold:", fold_index)
      test_indices = np.flatnonzero(self.folding.Fold==fold_index)
      train_indices = np.flatnonzero(self.folding.Fold!=fold_index)
      yield train_indices, test_indices

  def y(self):
    return np.asarray(self.folding.z)
  def X(self):
    return self
  def __nonzero__(self):
    return len(self.folding.index)>0
  def __len__(self):
    return len(self.folding.index)
  def __bool__(self):
    return len(self.folding.index)>0
  def __getitem__(self, key):
    if(self.verbose>0):
      if(isinstance(key, int) or isinstance(key, np.int64)):
        print("Query with key:", key)
      else:
        print("Query type:", type(key))
    if(isinstance(key, int) or isinstance(key, np.int64)):
      r, c = self.folding.iloc[key].Row, self.folding.iloc[key].Column
      r, c = r+self.kernel_size, c+self.kernel_size
      a_slice = self.slicer.apply_mask(row=r, column=c)
      return np.transpose(a_slice, axes=(2,1,0))
    if(isinstance(key, np.ndarray)):
      return np.asarray([self[i] for i in key], dtype=np.float64)
    else:
      print("Don't know how to handle ", type(key))
      return None

  def get_n_splits(self):
    return self.n_splits



from tensorflow.keras.layers import AveragePooling2D, Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def jigsaw_m( input_net, first_layer = None ):
    conv1 = Conv2D(128, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(input_net)
    jigsaw_t1_1x1 = Conv2D(256, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    jigsaw_t1_3x3_reduce = Conv2D(96, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    jigsaw_t1_3x3 = Conv2D(128, (3,3), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_3x3")(jigsaw_t1_3x3_reduce)
    jigsaw_t1_5x5_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    jigsaw_t1_5x5 = Conv2D(128, (5,5), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_5x5")(jigsaw_t1_5x5_reduce)
    jigsaw_t1_7x7_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    jigsaw_t1_7x7 = Conv2D(128, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_7x7")(jigsaw_t1_7x7_reduce)
    jigsaw_t1_9x9_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(conv1)
    jigsaw_t1_9x9 = Conv2D(64, (7,7), padding='same', activation = 'relu', kernel_regularizer = l2(0.002), name="i_9x9")(jigsaw_t1_9x9_reduce)
    jigsaw_t1_pool = MaxPooling2D(pool_size=(3,3), strides = (1,1), padding='same')(conv1)
    jigsaw_t1_pool_proj = Conv2D(32, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(jigsaw_t1_pool)
    if first_layer is None:
        jigsaw_t1_output = Concatenate(axis = -1)([jigsaw_t1_1x1, jigsaw_t1_3x3, jigsaw_t1_5x5,
                                                      jigsaw_t1_7x7, jigsaw_t1_9x9, jigsaw_t1_pool_proj])
    else:
        jigsaw_t1_first = Conv2D(96, (1,1), padding='same', activation = 'relu', kernel_regularizer = l2(0.002))(first_layer)
        jigsaw_t1_output = Concatenate(axis = -1)([jigsaw_t1_first, jigsaw_t1_1x1, jigsaw_t1_3x3,
                                                      jigsaw_t1_5x5, jigsaw_t1_7x7, jigsaw_t1_9x9,
                                                      jigsaw_t1_pool_proj])
    return jigsaw_t1_output

def jigsaw_m_end( input_net, num_classes = NUM_CLASSES, first_layer = None ):
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
    # my_input = Input( shape=IMAGE_DIMS, batch_shape=BATCH_DIMS )
my_input = Input( shape=IMAGE_DIMS )
# One jigsaw module(s)
jigsaw_01 = jigsaw_m( my_input )
# Attaches end to jigsaw modules, returns class within num_classes
loss3_classifier_act = jigsaw_m_end(jigsaw_01,
                                    num_classes = num_classes,
                                    first_layer = my_input ) # testing num_classes
# Builds model
model3 = Model( inputs = my_input, outputs = [loss3_classifier_act] )
#model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model3.summary()
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import rasterio as rio
from matplotlib import pyplot

tif_file = rio.open(tif_file_name)

### Come here
rSpCV = RasterSpCV(tif_file, kernel_size = kernel_pixels, sample=num_samples, random_state=42, verbose=0)
cv = rSpCV.SpatialCV_split()
X = rSpCV.X()
y = rSpCV.y()

cv_score = []
for i, (train, test) in enumerate(cv):
    print("Size (train, test)", len(train), len(test))
    # print("test:", test)
    # print("trainY:", y[train])
    model_2 = Model( inputs = my_input, outputs = [loss3_classifier_act] )
    model_2.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    #model3.summary()

    params = {'dim':(kernel_pixels,kernel_pixels), 'batch_size': batch_size,
            'n_classes': num_classes,
            'n_channels': image_channels,
            'shuffle': True, 'augment_data': augment_data}
    # construct the image generator for data augmentation
    y_binary = to_categorical(y[train], num_classes=num_classes)
    my_batch_gen = DataGenerator(X[train], y_binary, **params)
    early_stop = EarlyStopping( monitor = 'loss',
                                min_delta=0.01,
                                mode='auto', verbose=1, # mode was 'min'
                                patience=5)
    early_stop2= EarlyStopping( monitor = 'accuracy',
                                min_delta=0.01,
                                mode='auto', verbose=1, # mode was 'min'
                                patience=5)

    print("Running Fold", i+1)
    # model_2.fit(X[train], y[train], epochs=num_epochs, batch_size=batch_size)
    model_2.fit(my_batch_gen, epochs=num_epochs,
                callbacks = [early_stop, early_stop2],
                batch_size=batch_size)
    y_binary_test = to_categorical(y[test], num_classes=num_classes)
    result = model_2.evaluate(X[test], y_binary_test)
    # if we want only the accuracy metric
    cv_score.append(result[1])
    # we have to clear previous model to reset weights
    # currently keras doesn't have like model.reset()
    keras.backend.clear_session()

print("\nMean accuracy of the cross-validation: {}".format(np.mean(cv_score)))
print("\nMax accuracy of the cross-validation: {}".format(np.max(cv_score)))
print("\nMin accuracy of the cross-validation: {}".format(np.min(cv_score)))
print("\nStdDev accuracy of the cross-validation: {}".format(np.std(cv_score)))
