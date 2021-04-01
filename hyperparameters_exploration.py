import numpy as np
import math_utils as utils
import raw_images_manipulation_utilities as imanip
import os
import tifffile as tif
import matplotlib.pyplot as plt
import time
# define a procedure to find:
#   - shifts from the cameras
#   - sigma for merging
#   - mean 3D PSF extraction and saving


if __name__=='__main__':

    # shifts from cameras

    VOLUME_SLICES = 141
    TOTAL_VOLUMES =  1
    TOTAL_IMAGES = TOTAL_VOLUMES * VOLUME_SLICES
    IMAGES_DIMENSION = 1024
    BACKGROUND_FOLDER = '/Users/ngc/Desktop/test_grad/' # add a "/" at the eend
    IMAGES_FOLDER = '/Users/ngc/Desktop/test_grad/'# or os.getcwd()
    VOLUMES_OUTPUT_FOLDER = '/Users/ngc/Desktop/test_grad/alive_vol/' 
    DATA_OUTPUT_FOLDER = '/Users/ngc/Desktop/test_grad/alive_vol/' 

    if os.path.isdir(VOLUMES_OUTPUT_FOLDER) == False:
        os.mkdir(VOLUMES_OUTPUT_FOLDER)
    if os.path.isdir(DATA_OUTPUT_FOLDER) == False:
        os.mkdir(DATA_OUTPUT_FOLDER)
    
    filename_front = IMAGES_FOLDER + 'front.stack'
    filename_back = IMAGES_FOLDER + 'back.stack'
    front = utils.open_binary_volume_with_hotpixel_correction(filename_front,
                                                              VOLUME_SLICES,
                                                              IMAGES_DIMENSION)
    back = utils.open_binary_volume_with_hotpixel_correction(filename_back,
                                                              VOLUME_SLICES,
                                                              IMAGES_DIMENSION)
    front=front[:, :, ::-1]

    start=time.time()
    shifts = imanip.explore_cameras_offset(front, back)
    front_r = imanip.cameras_shift_registration(front, shifts)
    print(time.time() - start)
    print(shifts)

    merged = imanip.merge_views(front_r, back, 'gradient', sigma=40)
    merged.tofile(VOLUMES_OUTPUT_FOLDER + 'merged_gradient_median.stack')