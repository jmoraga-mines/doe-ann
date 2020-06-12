# -*- coding: utf-8 -*-
"""
Created on 2020-06-03

@authors: jim
"""

import numpy as np
from osgeo import gdal_array


def frame_image( image, frame_size = None ):
    if frame_size is None: return image
    assert( frame_size > 0 )
    i_c = image.shape[0]
    i_x = image.shape[1]
    i_y = image.shape[2]
    frame_c = i_c
    frame_x = i_x + 2*frame_size
    frame_y = i_y + 2*frame_size
    framed_img = np.zeros((frame_c, frame_x, frame_y))
    framed_img[:, frame_size:frame_size+i_x, frame_size:frame_size+i_y] = image
    return framed_img

def read_gdal_file( image_name ):
    '''read_gdal_file
    string image_name: name of an image file in GeoTiff format

    This function reads a GeoTIFF file, where each layer
    corresponds to a band.

    returns: a numpy array with 3 dimensions (width, height, channels)

    :param image_name:
    :return:
    '''
    # Read raster data as numeric array from file
    my_image = gdal_array.LoadFile(image_name)         # GDAL extracts image as (bands, height, width)
    # no_data = my_image.GetNoDataValue()
    # my_image = np.ma.masked_equal(my_image, no_data) # masks unavailable data from GeoTiff
    my_image = np.ma.masked_invalid(my_image)          # masks unavailable data from GeoTiff
    my_image = np.transpose(my_image, axes=(2,1,0))    # changes array to fit (width, height, channels)
    return my_image


def preprocess_sentinel_image(sentinel_image):
    '''preprocess_sentinel_image
        string sentinel_image: name of an image file in tiff format

        This function reads a sentinel-2 TIFF file, where each layer
        corresponds to a band. Output is a calibrated image

        returns: a numpy array with 3 dimensions (width, height, channels)
    '''
    my_image = np.array(sentinel_image).astype(np.float32)
    my_image[my_image>10000] = 10000.0
    my_image = my_image/10000.0
    return my_image

