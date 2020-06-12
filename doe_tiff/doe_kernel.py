import numpy as np


class Kernel3D:
    def __init__(self, cols=3, rows=3, shape='rect', radius=None):
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
        self.mask = self.mask[:, :, np.newaxis]
        assert((rows%2) == 1)
        assert((cols%2) == 1)

    def round_mask(self, radius):
        diameter = 2*radius+1
        mask = np.zeros((diameter, diameter))
        sq_radius = radius**2
        for i in range(diameter):
            for j in range(diameter):
                if ((i-radius)**2+(j-radius)**2) <= sq_radius:
                    mask[i, j] = 1
        return mask

    def getSubset(self, matrix, column, row):
        m_rows = matrix.shape[1]
        assert((row >= self.row_buffer) and (row < (m_rows-self.row_buffer)))
        m_cols = matrix.shape[0]
        assert((column >= self.col_buffer) and (column < (m_cols-self.col_buffer)))
        row_start = row-self.row_buffer
        row_end = row+self.row_buffer
        column_start = column-self.col_buffer
        column_end = column+self.col_buffer
        small_matrix = matrix[column_start:column_end+1, row_start:row_end+1, :]
        return small_matrix*self.mask

    def getPercentage(self, matrix, column, row):
        test_matrix = self.getSubset(matrix, column, row)
        return test_matrix.mean()


class GeoTiffConvolution:
    def __init__(self, land_matrix, kernel_cols=None, kernel_rows=3,
                 kernel_shape='rect', kernel_radius=0):
        if kernel_cols is None:
            kernel_cols = kernel_rows
        assert(kernel_cols < land_matrix.shape[0])
        assert(kernel_rows < land_matrix.shape[1])
        assert((kernel_shape == 'rect') or (kernel_shape == 'circle'))
        if kernel_shape == 'rect':
            self.kernel = Kernel3D(rows=kernel_rows, cols=kernel_cols)
        else:
            self.kernel = Kernel3D(radius=kernel_radius, shape=kernel_shape)
            kernel_rows = kernel_cols = 2*kernel_radius+1
        self.kernel_rows = kernel_rows
        self.kernel_cols = kernel_cols
        self.land_matrix = land_matrix
        self.land_matrix_cols = land_matrix.shape[0]
        self.land_matrix_rows = land_matrix.shape[1]
        self.land_matrix_channels = land_matrix.shape[2]
        self.small_row_min = self.kernel.row_buffer
        self.small_row_max = self.land_matrix_rows - self.small_row_min
        self.small_column_min = self.kernel.col_buffer
        self.small_column_max = self.land_matrix_cols - self.small_column_min

    def apply_mask(self, column, row):
        return self.kernel.getSubset(self.land_matrix, column, row)

    def calculate(self):
        m1 = np.zeros_like(self.land_matrix, dtype='float')
        for j in range(self.small_row_min, self.small_row_max):
            for i in range(self.small_column_min, self.small_column_max):
                m1[i, j] = self.kernel.getPercentage(self.land_matrix, i, j)
        return m1
