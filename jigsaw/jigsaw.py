from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adadelta #, Adam, Nadam, Adagrad, Adamax, SGD
import numpy as np
from skimage.transform import rotate


# Helper class, generates data for keras
class DataGenerator(Sequence): 
    'Generates data for Keras'
    def __init__(self, data_set, labels, batch_size,
            dim, # (KERNEL_PIXELS,KERNEL_PIXELS,CHANNELS)
            n_channels, n_classes,
            shuffle=True, augment_data=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_set = data_set
        self.n_channels = n_channels
        #assert (n_channels > dim.shape[2]), f"Number of channels ({n_channels}) do not match image definition"
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
                image_raw = rotate(image_raw, angle=15*np.random.randint(6), mode='symmetric')
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


def jigsaw_m( input_net, first_layer = None , internal_size = 13):
    conv1 = Conv2D(128, (1,1), padding='same', activation = 'relu', 
                   kernel_regularizer = l2(0.002))(input_net)
    jigsaw_t1_1x1 = Conv2D(256, (1,1), padding='same', activation = 'relu', 
                           kernel_regularizer = l2(0.002))(conv1)
    jigsaw_t1_3x3_reduce = Conv2D(96, (1,1), padding='same', activation = 'relu', 
                                  kernel_regularizer = l2(0.002))(input_net)
    jigsaw_t1_3x3 = Conv2D(128, (3,3), padding='same', activation = 'relu', 
                           kernel_regularizer = l2(0.002), name="i_3x3")(jigsaw_t1_3x3_reduce)
    if (internal_size >= 5):
        jigsaw_t1_5x5_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', 
                                      kernel_regularizer = l2(0.002))(input_net)
        jigsaw_t1_5x5 = Conv2D(128, (5,5), padding='same', activation = 'relu', 
                               kernel_regularizer = l2(0.002), name="i_5x5")(jigsaw_t1_5x5_reduce)
    if (internal_size >= 7):
        jigsaw_t1_7x7_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', 
                                      kernel_regularizer = l2(0.002))(input_net)
        jigsaw_t1_7x7 = Conv2D(128, (7,7), padding='same', activation = 'relu', 
                               kernel_regularizer = l2(0.002), name="i_7x7")(jigsaw_t1_7x7_reduce)
    if (internal_size >= 9):
        jigsaw_t1_9x9_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', 
                                      kernel_regularizer = l2(0.002))(input_net)
        jigsaw_t1_9x9 = Conv2D(64, (9,9), padding='same', activation = 'relu', 
                               kernel_regularizer = l2(0.002), name="i_9x9")(jigsaw_t1_9x9_reduce)
    if (internal_size >= 11):
        jigsaw_t1_11x11_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', 
                                        kernel_regularizer = l2(0.002))(input_net)
        jigsaw_t1_11x11 = Conv2D(64, (11,11), padding='same', activation = 'relu', 
                                 kernel_regularizer = l2(0.002), name="i_11x11")(jigsaw_t1_11x11_reduce)
    if (internal_size >= 13):
        jigsaw_t1_13x13_reduce = Conv2D(16, (1,1), padding='same', activation = 'relu', 
                                        kernel_regularizer = l2(0.002))(input_net)
        jigsaw_t1_13x13 = Conv2D(64, (13,13), padding='same', activation = 'relu', 
                                 kernel_regularizer = l2(0.002), name="i_13x13")(jigsaw_t1_13x13_reduce)
    jigsaw_t1_pool = MaxPooling2D(pool_size=(3,3), strides = (1,1), padding='same')(conv1)
    jigsaw_t1_pool_proj = Conv2D(32, (1,1), padding='same', activation = 'relu', 
                                 kernel_regularizer = l2(0.002))(jigsaw_t1_pool)
    jigsaw_list = [jigsaw_t1_1x1, jigsaw_t1_3x3]
    if (internal_size >= 5):
        jigsaw_list.append(jigsaw_t1_5x5)
    if (internal_size >= 7):
        jigsaw_list.append(jigsaw_t1_7x7)
    if (internal_size >= 9):
        jigsaw_list.append(jigsaw_t1_9x9)
    if (internal_size >= 11):
        jigsaw_list.append(jigsaw_t1_11x11)
    if (internal_size >= 13):
        jigsaw_list.append(jigsaw_t1_13x13)
    jigsaw_list.append(jigsaw_t1_pool_proj)
    if first_layer is not None:
        jigsaw_t1_first = Conv2D(96, (1,1), padding='same', activation = 'relu', 
                                 kernel_regularizer = l2(0.002))(first_layer)
        jigsaw_list.append(jigsaw_t1_first)
    jigsaw_t1_output = Concatenate(axis = -1)(jigsaw_list)
    return jigsaw_t1_output

def jigsaw_m_end( input_net, num_classes, first_layer = None ):
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


# Builds model
def build_jigsaw(internal_size=13, num_classes=2, image_dim = (19, 19, 7), verbose=1):
    my_input = Input( shape=image_dim )
    if(verbose>0):
        print(f"*** Building Jigsaw with up to {internal_size}x{internal_size} kernels")
    # One jigsaw module(s)
    jigsaw_01 = jigsaw_m( my_input, internal_size = internal_size )
    # Attaches end to jigsaw modules, returns class within num_classes
    loss3_classifier_act = jigsaw_m_end(jigsaw_01,
                                    num_classes = num_classes,
                                    first_layer = my_input ) # testing num_classes
    model3 = Model( inputs = my_input, outputs = loss3_classifier_act )
    model3.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    return model3

if __name__ == "__main__":
    from tensorflow.keras.models import clone_model
    from tensorflow.keras.utils import plot_model
    import gc
    
    gc.collect()

    m = clone_model(build_jigsaw(internal_size = 11, 
                                 num_classes = 2,
                                 image_dim = (20, 40, 9)))
    m.summary()
    
    # Plots architecture
    plot_model(m)
