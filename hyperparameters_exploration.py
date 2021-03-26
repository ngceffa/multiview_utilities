import numpy as np
import math_utils as utils
import raw_images_manipulation_utilities as imanip
import os
import tifffile as tif
import matplotlib.pyplot as plt
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
    
    # filename_front = 'front.stack'
    # filename_back = 'back.stack'
    # front = utils.open_binary_volume_with_hotpixel_correction(filename_front,
    #                                                           VOLUME_SLICES,
    #                                                           IMAGES_DIMENSION)
    # back = utils.open_binary_volume_with_hotpixel_correction(filename_back,
    #                                                           VOLUME_SLICES,
    #                                                           IMAGES_DIMENSION)
    
    front = tif.imread(IMAGES_FOLDER + 'SPC00_TM00000_ANG000_CM0_CHN00_PH0.tif')
    back = tif.imread(IMAGES_FOLDER + 'SPC00_TM00000_ANG000_CM1_CHN00_PH0.tif')
    images = np.arange(75, 80, 2)
    front=front[:, :, ::-1]
    shifts = imanip.explore_cameras_offset(front, back, images)
    front_r = imanip.cameras_shift_registration(front, shifts)
    print(shifts)
    fig=plt.figure('1')
    fig.add_subplot(311)
    plt.imshow(front[80,:,:])
    fig.add_subplot(312)
    plt.imshow(front_r[80,:,:])
    fig.add_subplot(313)
    plt.imshow(back[80,:,:])
    plt.show()