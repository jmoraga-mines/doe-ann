import numpy as np


class Kernel3D:
    def __init__(self, rows=3, cols=3, shape='rect', radius=None):
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
        assert((rows%2) == 1)
        assert((cols%2) == 1)

    def round_mask(self, radius):
        diameter = 2*radius+1
        mask = np.zeros((diameter, diameter))
        sq_radius = radius**2
        for i in range(diameter):
            for j in range(diameter):
                if ((i-radius)**2+(j-radius)**2)<=sq_radius:
                    mask[i,j]=1
        return mask

    def getSubset(self, matrix, x, y):
        m_rows = matrix.shape[1]
        assert((x >= self.row_buffer) and (x < (m_rows-self.row_buffer)))
        m_cols = matrix.shape[2]
        assert((y >= self.col_buffer) and (y < (m_cols-self.col_buffer)))
        x_start = x-self.row_buffer
        x_end = x+self.row_buffer
        y_start = y-self.col_buffer
        y_end = y+self.col_buffer
        small_matrix = matrix[:, x_start:x_end+1, y_start:y_end+1]
        return small_matrix*self.mask

    def getPercentage(self, matrix, x, y):
        test_matrix = self.getSubset(matrix, x, y)
        return test_matrix.mean()


class GeoTiffConvolution:
    def __init__(self, land_matrix, kernel_rows=3,
                 kernel_cols=None, kernel_shape='rect', kernel_radius=0):
        if kernel_cols is None:
            kernel_cols = kernel_rows
        assert(kernel_rows < land_matrix.shape[1])
        assert(kernel_cols < land_matrix.shape[2])
        assert((kernel_shape == 'rect') or (kernel_shape == 'circle'))
        if kernel_shape == 'rect':
            self.kernel = Kernel3D(rows=kernel_rows, cols=kernel_cols)
        else:
            self.kernel = Kernel3D(radius=kernel_radius, shape=kernel_shape)
            kernel_rows = kernel_cols = 2*kernel_radius+1
        self.kernel_rows = kernel_rows
        self.kernel_cols = kernel_cols
        self.land_matrix = land_matrix
        self.land_matrix_channels = land_matrix.shape[0]
        self.land_matrix_rows = land_matrix.shape[1]
        self.land_matrix_cols = land_matrix.shape[2]
        self.small_xmin = self.kernel.row_buffer
        self.small_xmax = self.land_matrix_rows - self.small_xmin
        self.small_ymin = self.kernel.col_buffer
        self.small_ymax = self.land_matrix_cols - self.small_ymin

    def apply_mask(self, x, y):
        return self.kernel.getSubset(self.land_matrix, x, y)

    def calculate(self):
        m1 = np.zeros_like(self.land_matrix, dtype='float')
        for i in range(self.small_xmin, self.small_xmax):
            for j in range(self.small_ymin, self.small_ymax):
                m1[i, j] = self.kernel.getPercentage(self.land_matrix, i, j)
        return m1
