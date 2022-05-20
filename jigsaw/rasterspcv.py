import numpy as np
from sklearn.cluster import MiniBatchKMeans # , KMeans
from sklearn.model_selection import BaseCrossValidator
from abc import  ABC #, abstractmethod, ABCMeta
import pandas as pd
import geopandas as gpd
import rasterio as rio
import random


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

class RasterSpCV(BaseCrossValidator, list, ABC):
    def __init__(self, base_image, kernel_size, num_channels,
                 no_value = np.NaN, verbose = 0,
                 partitions = 5,
                 augment = False,
                 sample = None, 
                 random_state = None):
        if(isinstance(base_image, str)):
            try:
                base_image = rio.open(base_image)
            except:
                print(f"Failed to open file {base_image}")
        assert isinstance(base_image, rio.io.DatasetReader), f"Wrong type, received {type(base_image)}"
        try:
            c, h, w = base_image.count, base_image.height, base_image.width
        except:
            c=h=w=0
        assert (c>1) and (h>kernel_size*2) and  (w>kernel_size*2)
        assert ((isinstance(sample, type(None))) or (isinstance(sample, int))), f"Wrong type, received {type(sample)}"
        assert ((kernel_size>0) and ((kernel_size%2)==1))
        self.augment = augment
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
        if(verbose>1):
            print("Reading raster file")
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
        if(not isinstance(sample, type(None))):
            df=df.sample(n = sample, random_state=random_state)
        if(verbose>1):
            print("Transforming coordinates to (x, y)")
        geometry = gpd.points_from_xy(df.x, df.y)
        tif_crs = base_image.crs
        gdf = gpd.GeoDataFrame(df, crs=tif_crs, geometry=geometry) # [['z', 'Column', 'Row']]
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
        _land_matrix = _land_matrix[1:(num_channels+1), :, :]
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
        
    def RepeatedSpCV(self, n_repeats=2):
        for i in range(n_repeats):
            for fold_index in self.partitions:
                if(self.verbose>1):
                    print("Fold:", fold_index)
                test_indices = np.flatnonzero(self.folding.Fold==fold_index)
                train_indices = np.flatnonzero(self.folding.Fold!=fold_index)
                random.shuffle(train_indices)
                random.shuffle(test_indices)
                yield (train_indices), (test_indices)


    def SpatialCV(self):
        for fold_index in self.partitions:
            if(self.verbose>1):
                print("Fold:", fold_index)
            test_indices = np.flatnonzero(self.folding.Fold==fold_index)
            train_indices = np.flatnonzero(self.folding.Fold!=fold_index)
            random.shuffle(train_indices)
            random.shuffle(test_indices)
            yield (train_indices), (test_indices)

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
        if(self.verbose>2):
            if(isinstance(key, int) or isinstance(key, np.int64)):
                print("Query with key:", key)
            else:
                print("Query type:", type(key))
        if(isinstance(key, int) or isinstance(key, np.int64)):
            r, c = self.folding.iloc[key].Row, self.folding.iloc[key].Column
            r, c = r+self.kernel_size, c+self.kernel_size
            a_slice = self.slicer.apply_mask(row=r, column=c)
            a_slice = np.transpose(a_slice, axes=(2,1,0))
            if (self.augment):
                a_slice = rotate(a_slice, angle=15*np.random.randint(6), mode='symmetric')
                # a_slice = np.rot90(a_slice, k= np.random.randint(3))
                if np.random.randint(1):
                    a_slice = np.fliplr(a_slice)
                if np.random.randint(1):
                    a_slice = np.flipud(a_slice)
            return a_slice
        if(isinstance(key, np.ndarray)):
            return np.asarray([self[i] for i in key], dtype=np.float64)
        print("Don't know how to handle ", type(key))
        return None

    def get_n_splits(self):
        return self.n_splits

    
    
if __name__ == "__main__":
    rSpCV = RasterSpCV('../brady_ai_stack.grd', 
                       kernel_size = 19, 
                       num_channels = 7,
                       sample = 10, 
                       verbose=2)
    cv = rSpCV.SpatialCV_split()
    for tr, ts in cv:
        pass
    print(rSpCV.slicer.land_matrix.shape)

