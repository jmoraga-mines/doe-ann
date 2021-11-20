# -*- coding: utf-8 -*-
"""
  Create Dataset directories from DOE Geothermal
  layer input GeoTIFFs. They have the following layers:
  - Minerals
  - Faults
  - Land Surface Temperature
  - Geothermal Presence (Ground Truth)

  Created 2020-06-10
  Updated 2020-06-12

  @authors: Jim Moraga <jmoraga@mines.edu>
"""

import numpy as np
import doe_tiff as dt
import os
import argparse
from tqdm import tqdm


# Default values
SAMPLES_TO_CREATE = 1200
KERNEL_PIXELS = 17
KERNEL_CHANNELS = 5
DEFAULT_DIRECTORY = 'dataset'


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input multi-band image (i.e., image file name)")
    ap.add_argument("-c", "--channels", required=False,
                    help="number of channels/bands to extract",
                    type=int, default=KERNEL_CHANNELS)
    ap.add_argument("-d", "--directory", required=False,
                    help="path to directory where samples will be written to",
                    default=DEFAULT_DIRECTORY)
    ap.add_argument("-s", "--samples", required=False, help="number of samples to create",
                    type=int, default=SAMPLES_TO_CREATE)
    ap.add_argument("-k", "--kernel_size", required=False,
                    help="number of pixels per side in the kernel (use an odd number)",
                    type=int, default=KERNEL_PIXELS)
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = parse_arguments()
    image_name = args["image"]
    num_channels = args["channels"]
    output_directory = args["directory"]
    num_samples = args["samples"]
    kernel_size = args["kernel_size"]
    PADDING = int(kernel_size/2)
    img_b = dt.io.read_gdal_file(image_name)
    max_channels = img_b.shape[2]
    assert num_channels>0, 'Channels has to be a positive integer'
    assert num_channels<=max_channels, 'Channels has to be equal or lower than {}'.format(max_channels)
    img_b_scaled = img_b[:, :, 1:num_channels+1]
    mask_b = img_b[:, :, 0] # first band is Ground Truth
    for i in range(0, img_b_scaled.shape[2]):
        print("band (",i,") min:", img_b_scaled[:,:,i].min())
        print("band (",i,") max:", img_b_scaled[:,:,i].max())
    img_b_scaled = dt.frame_image(img_b_scaled, PADDING)
    sc = dt.GeoTiffConvolution(img_b_scaled, kernel_size, kernel_size)
    # Land classes go from 0 to 1
    for land_type_class in range(0, 2):
        mask_b_class = np.array(np.where(mask_b == land_type_class)).T
        print('Items in class ', land_type_class, ': ', len(mask_b_class))
        mask_b_scaled = mask_b_class+[PADDING, PADDING]
        choices = np.random.choice(len(mask_b_scaled), num_samples, replace=False)
        path_name = output_directory + '/' + str(land_type_class) + '/'
        os.makedirs(path_name)
        for c in tqdm(choices, desc="Calculating...", ascii=False, ncols=75):
            my_slice = sc.apply_mask(mask_b_scaled[c][0], mask_b_scaled[c][1])
            file_name = path_name + 'slice_'+str(c)+'.npy'
            # Save model file
            f = open(file_name, 'wb')
            np.save(f, my_slice)
