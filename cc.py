# -*- coding: utf-8 -*-
"""
  Create maps showing training data points
  the "Selected" layer includes training, testing and validation sets

  Created 2021-05-24
  Updated 2021-05-25

  @authors: Jim Moraga <jmoraga@mines.edu>
"""

import numpy as np
import doe_tiff as dt
import os
import argparse
from osgeo import gdal
from osgeo.gdalconst import *


# Default values
SAMPLES_TO_CREATE = 100000
KERNEL_PIXELS = 27
KERNEL_CHANNELS = 3
DEFAULT_DIRECTORY = 'dataset'


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input multi-band image (i.e., image file name)")
    ap.add_argument("-c", "--channels", required=False,
                    help="number of channels/bands to extract",
                    type=int, default=KERNEL_CHANNELS)
    ap.add_argument("-s", "--samples", required=False, help="number of samples to create",
                    type=int, default=SAMPLES_TO_CREATE)
    ap.add_argument("-k", "--kernel_size", required=False,
                    help="number of pixels per side in the kernel (use an odd number)",
                    type=int, default=KERNEL_PIXELS)
    ap.add_argument("-o", "--output_raster", required=False, type=str, help="Raster file ending in .gri",
                    default = "training_map.gri")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = parse_arguments()
    image_name = args["image"]
    num_channels = args["channels"]
    num_samples = args["samples"]
    kernel_size = args["kernel_size"]
    output_raster = args["output_raster"]

    PADDING = int(kernel_size/2)
    img_b = dt.io.read_gdal_file(image_name)
    max_channels = img_b.shape[2]
    assert num_channels>0, 'Channels has to be a positive integer'
    assert num_channels<=max_channels, 'Channels has to be equal or lower than {}'.format(max_channels)
    img_b_scaled = img_b[:, :, 1:num_channels+1]
    mask_b = img_b[:, :, 0] # first band is Ground Truth
    max_x, max_y = mask_b.shape
    mask_c = np.zeros_like(mask_b) # Create new band to host the random choices
    mask_t = np.zeros_like(mask_b) # Create new band to host the training choices
    mask_a = np.zeros_like(mask_b) # Create new band to host the training area
    # Land classes go from 0 to 1
    for land_type_class in range(0, 2):
        mask_b_class = np.array(np.where(mask_b == land_type_class)).T
        print('Items in class ', land_type_class, ': ', len(mask_b_class))
        mask_b_scaled = mask_b_class
        choices = np.random.choice(len(mask_b_class), num_samples, replace=False)
        for c in choices:
            mask_c[mask_b_scaled[c][0], mask_b_scaled[c][1]] = 1
        choices2 = np.random.choice(len(choices), int(num_samples*0.2), replace=False)
        for c2 in choices2:
            c = choices[c2]
            x = mask_b_scaled[c][0]
            x_left = max(x-PADDING, 0)
            x_right = min(x+PADDING, max_x)
            y = mask_b_scaled[c][1]
            y_top = max(y-PADDING, 0)
            y_bottom = min(y+PADDING, max_y)
            mask_t[x, y] = 1
            mask_a[x_left:x_right, y_top:y_bottom] = 1
    inDs = gdal.Open(image_name)
    band1 = inDs.GetRasterBand(1)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    cropData = band1.ReadAsArray(0,0,cols,rows)
    driver = inDs.GetDriver()
    outDs = driver.Create(output_raster, cols, rows, 4, GDT_Int32)
    outBand_1 = outDs.GetRasterBand(1)
    outBand_2 = outDs.GetRasterBand(2)
    outBand_3 = outDs.GetRasterBand(3)
    outBand_4 = outDs.GetRasterBand(4)
    mask_b = np.where(~np.logical_or(mask_b==0, mask_b==1), np.nan, mask_b)
    outBand_1.WriteArray(np.asarray(mask_b).T)
    outBand_1.SetNoDataValue(-99)
    outBand_1.SetDescription("Geothermal")
    outBand_1.FlushCache()
    outBand_2.WriteArray(np.asarray(mask_c).T)
    outBand_2.SetNoDataValue(-99)
    outBand_2.SetDescription("Selected")
    outBand_2.FlushCache()
    outBand_3.WriteArray(np.asarray(mask_t).T)
    outBand_3.SetNoDataValue(-99)
    outBand_3.SetDescription("Training")
    outBand_3.FlushCache()
    outBand_4.WriteArray(np.asarray(mask_a).T)
    outBand_4.SetNoDataValue(-99)
    outBand_4.SetDescription("Training_Area")
    outBand_4.FlushCache()
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())
    print("saving file: ", output_raster)
    outDs.FlushCache()
    outDs = None


