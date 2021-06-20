import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import importlib
import os
import sys
# local folder with homemade libraries
sys.path.append('/home/ngc/Documents/GitHub/multiview_utilities')
# HOMEMADE LIBRARIES AND IMPORTS
import math_utils as matut
import raw_images_manipulation_utilities as rim
importlib.reload(matut);
importlib.reload(rim); # reload the modules for updating to eventual changes


if __name__=='__main__':

    # --------------------------------------------------------------------------
    #Local variables and constants.
    # Carefully read and change the variables in the next section
    # to encompass the data to be analysed.
    VOLUME_SLICES = 53 # number of images for each volume
    TIMEPOINTS =  1
    TOTAL_IMAGES = TIMEPOINTS * VOLUME_SLICES # total of raw data-images
    IMAGES_DIMENSION = 2304 # assumed square images. N.b. try to have 2^n pixels
    BACKGROUND_FOLDER = '/home/ngc/Data/16_06/fift_1_percent_agar_20210616_164944'
    RAW_SLICES_FOLDER = '/home/ngc/Data/16_06/fift_1_percent_agar_20210616_164944'
    VOLUMES_OUTPUT_FOLDER = \
        '/home/ngc/Data/16_06/fift_1_percent_agar_20210616_164944/output_test'
    # create output folder if not present:
    if os.path.isdir(VOLUMES_OUTPUT_FOLDER) == False:
        os.mkdir(VOLUMES_OUTPUT_FOLDER)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # list of names of all the volumes as 
    # [timepoint (int), vol_cam_0 [str], vol_cam_1 (str)]
    # next function uses default file names as saved by Labview software
    # (check arguments in "rim" library)
    data_list = rim.file_names_list(TIMEPOINTS)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Camera offset compensation
    # The cameras are not pixel-exactly aligned: the offset in their views
    # can be calculated with a 2D cross-correlation.
    # This process returns a 2D displacement vector, 
    # that will be addedd (during image processing) to CAM_01 
    # to have a better superpositoin with CAM_00.
    # In the next cell:

    # 1. Select a volume, and in that volume a (few) slice(s) 
    #    that will be used to find the 2D mismatch.

    # 2. A pictorial result will tell if it worked. 
    #    It should consist of: two superimposed imaged before displacement, 
    #    after displacement, and a print() of the displacement vector.
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    timepoint_for_offset_calculation = 0
    slices_for_offset_calculation = np.asarray((20, 25, 30, 35, 40, 45))
    extent = 1000 # pixels used to looka at a central subregion in the images
    centre = IMAGES_DIMENSION / 2
    shift_row, shift_col = 0, 0

    stack_0 = rim.open_binary_stack(
        RAW_SLICES_FOLDER + data_list[timepoint_for_offset_calculation][1],
        BACKGROUND_FOLDER + '/Background_0.tif',
        size_x=IMAGES_DIMENSION,
        size_y=IMAGES_DIMENSION)
    stack_1 = rim.open_binary_stack(
        RAW_SLICES_FOLDER + data_list[timepoint_for_offset_calculation][2],
        BACKGROUND_FOLDER + '/Background_1.tif',
        size_x=IMAGES_DIMENSION,
        size_y=IMAGES_DIMENSION)
    
    for i in slices_for_offset_calculation:
        image_0 = stack_0[i, :, :]
        image_1 = stack_1[i, :, :]
    
        image_0_b = image_0[int(centre - extent/2):int(centre + extent/2),
                            int(centre - extent/2):int(centre + extent/2)]
        image_1_b = image_1[int(centre - extent/2):int(centre + extent/2),
                            int(centre - extent/2):int(centre + extent/2)]
        # n.b. images are one the mirror of the other, hence the [:, ::-1]
        shifts = rim.find_camera_registration_offset(image_0_b,
                                image_1_b[:, ::-1])
        shift_row += shifts[0] / len(slices_for_offset_calculation)
        shift_col += shifts[1] / len(slices_for_offset_calculation)

    shift = np.asarray((shift_row, shift_col)).astype(np.int_)
    new_image= matut.shift_image(image_1[:,::-1], shift)

    offset_recap = plt.figure('offset recap', figsize=(15, 7))
    offset_recap.add_subplot(121)
    plt.imshow(image_0, alpha=.7, cmap='Oranges')
    plt.imshow(image_1[:, ::-1], alpha=.5, cmap='Blues')
    offset_recap.add_subplot(122)
    plt.imshow(image_0, alpha=.7, cmap='Oranges')
    plt.imshow(new_image, alpha=.5, cmap='Blues')
    plt.show()

    print(f'\nAverage shift: {shift} ([row, cols] in pixels)\n')
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # Temporal (rigid) registration]

    # The sample suffers from a slow drift: it usually sinks.
    # In order to find the drift vector, an analysis similar to the previous
    # one is performed: a 3D cross-correlation between multiple couples
    # of volumes. The first is always the starting volume, the second is
    # as a function of time: it can be subtracted to each volume

    # In the next sequence:

    # 1. Select the step [a.u., "timesteps"] used to include volumes
    #    in the algorithm.

    # 2. A pictorial result will tell if it worked. 
    #    It should consist of 3 linear fits for the 3 displacement compontents.
    #    Only one component is expected to be substantially != 0.
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    timestep = 10 # analysis performed every "timestep" volumes
    extent_x_y = 1000
    centre = IMAGES_DIMENSION / 2
    shifts_row, shifts_col, shifts_plane = [], [], []

    # by default we use camera 0, but using camera 1 is also fine.
    # (just change filename and Background accordingly)

    start_stack = rim.open_binary_stack(
        RAW_SLICES_FOLDER + data_list[0][1],
        BACKGROUND_FOLDER + '/Background_0.tif',
        size_x=IMAGES_DIMENSION,
        size_y=IMAGES_DIMENSION)
    # Use only a central region
    start_focus = start_stack[
        :,
        int(centre - extent/2):int(centre + extent/2),
        int(centre - extent/2):int(centre + extent/2)
        ]
    
    
    for t in range(0, TIMEPOINTS, timestep):
        moving_stack = rim.open_binary_stack(
        RAW_SLICES_FOLDER + data_list[t][1],
        BACKGROUND_FOLDER + '/Background_0.tif',
        size_x=IMAGES_DIMENSION,
        size_y=IMAGES_DIMENSION)
        # Use only a central region
        moving_focus = moving_stack[
        :,
        int(centre - extent/2):int(centre + extent/2),
        int(centre - extent/2):int(centre + extent/2)
        ]
        
        shifts = rim.find_sample_drift(start_focus, moving_focus)
        shifts_plane.append(shift[0])
        shifts_row.append(shift[1])
        shifts_col.append(shift[2])
    
    # FITTING (linear)
    # y = m x + q

    # numpy.polyfit(x, y, deg,

    x_plane = np.arange(0, len(shifts_plane), 1)
    x_row = np.arange(0, len(shifts_row), 1)
    x_col = np.arange(0, len(shifts_col), 1)

    fit_plane = np.polyfit(x_plane, np.asarray(shifts_plane), 1)
    y_plane = fit_plane[0] * x_plane + fit_plane[1]

    fit_row = np.polyfit(x_row, np.asarray(shifts_row), 1)
    y_row = fit_row[0] * x_row + fit_row[1]

    fit_col = np.polyfit(x_col, np.asarray(shifts_col), 1)
    y_col = fit_col[0] * x_col + fit_col[1]

    summary = plt.figure('summary')
    summary.add_subplot(311)
    plt.plot(x_plane, shifts_plane, 'o', color='azure')
    plt.plot(x_plane, fit_plane, '--', color='blue', alpha=.4)
    summary.add_subplot(312)
    plt.plot(shifts_row, 'o', color='orange')
    plt.plot(x_row, fit_row, '--', color='red', alpha=.4)
    summary.add_subplot(313)
    plt.plot(shifts_col, 'o', color='lightgreen')
    plt.plot(x_col, fit_col, '--', color='green', alpha=.4)
    plt.show()

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # Feature extraction
    # TO merge images following Preibish et al. approximation to local entropy,
    # two sigmas must be selected. The first one should be larger than the linear
    # dimension of sensible features.
    # e.g. feature = 10 pixels, value = 15 pixels (???)
    # Find it with the cursos over the image,
    # and then save it in "sigma" value.
    #

    feature_dimension = plt.figure('extract feature')


     # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # Saving all the hyperparameters in a .csv
    # - cameras offset
    # - drift fit parameters
    # - sigma 