import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import importlib
import os
import sys
import csv
from scipy.optimize import curve_fit
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
    # to match the volume to be analysed.

    SLICES = 53 # number of images for each volume
    TIMEPOINTS =  161 # aka number of acquired volumes
    TOTAL_IMAGES = TIMEPOINTS * SLICES # total of raw data-images
    IMAGES_DIMENSION = 1024 # assumed square images.
        # N.b. try to have 2^n pixels: it speeds up FFt calculations.
    # Folder to get data and save data:
    RAW_SLICES_FOLDER = '/home/ngc/Data/560_other_larva'
    BACKGROUND_FOLDER = '/home/ngc/Data/16_06/fift_1_percent_agar_20210616_164944'
    VOLUMES_OUTPUT_FOLDER = '/home/ngc/Data/560_other_larva/output_test'
    PARAMS_OUTPUT_FOLDER = '/home/ngc/Data/560_other_larva'
    # create output folder if not present:
    if os.path.isdir(VOLUMES_OUTPUT_FOLDER) == False:
        os.mkdir(VOLUMES_OUTPUT_FOLDER)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Create a list of names of all the volumes as 
    # [timepoint (int), vol_cam_0 [str], vol_cam_1 (str)]
    # next function uses a default file names paradigm, 
    # as saved by Labview software
    # (check arguments in "raw_images_manipulation_utilities" library)
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

    # timepoint_for_offset_calculation = 0
    # slices_for_offset_calculation = np.asarray((10, 15, 22, 30, 35))
    # extent = 800 # pixels used to looka at a central subregion in the images
    # centre = IMAGES_DIMENSION / 2
    # shift_row, shift_col = 0, 0

    # stack_0 = rim.open_binary_stack(
    #     RAW_SLICES_FOLDER + data_list[timepoint_for_offset_calculation][1],
    #     BACKGROUND_FOLDER + '/Background_0.tif',
    #     size_x=IMAGES_DIMENSION,
    #     size_y=IMAGES_DIMENSION)
    # stack_1 = rim.open_binary_stack(
    #     RAW_SLICES_FOLDER + data_list[timepoint_for_offset_calculation][2],
    #     BACKGROUND_FOLDER + '/Background_1.tif',
    #     size_x=IMAGES_DIMENSION,
    #     size_y=IMAGES_DIMENSION)
    
    # for i in slices_for_offset_calculation:
    #     image_0 = stack_0[i, :, :]
    #     image_1 = stack_1[i, :, :]
    
    #     image_0_b = image_0[int(centre - extent/2):int(centre + extent/2),
    #                         int(centre - extent/2):int(centre + extent/2)]
    #     image_1_b = image_1[int(centre - extent/2):int(centre + extent/2),
    #                         int(centre - extent/2):int(centre + extent/2)]
    #     # n.b. images are one the mirror of the other, hence the [:, ::-1]
    #     shifts = rim.find_camera_registration_offset(image_0_b,
    #                             image_1_b[:, ::-1])
    #     shift_row += shifts[0] / len(slices_for_offset_calculation)
    #     shift_col += shifts[1] / len(slices_for_offset_calculation)

    # shift = np.asarray((shift_row, shift_col)).astype(np.int_)
    # new_image= matut.shift_image(image_1[:,::-1], shift)

    # offset_recap = plt.figure('offset recap', figsize=(15, 7))
    # offset_recap.add_subplot(121)
    # plt.imshow(image_0, alpha=.7, cmap='Oranges')
    # plt.imshow(image_1[:, ::-1], alpha=.5, cmap='Blues')
    # offset_recap.add_subplot(122)
    # plt.imshow(image_0, alpha=.7, cmap='Oranges')
    # plt.imshow(new_image, alpha=.5, cmap='Blues')
    # plt.show()

    # print(f'\nAverage shift: {shift} ([row, cols] in pixels)\n')
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

    timestep = 20 # analysis performed every "timestep" volumes
    extent_x_y = 800
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
        int(centre - extent_x_y/2):int(centre + extent_x_y/2),
        int(centre - extent_x_y/2):int(centre + extent_x_y/2)
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
        int(centre - extent_x_y/2):int(centre + extent_x_y/2),
        int(centre - extent_x_y/2):int(centre + extent_x_y/2)
        ]
        
        shift = rim.find_sample_drift(start_focus, moving_focus)
        shifts_plane.append(shift[0])
        shifts_row.append(shift[1])
        shifts_col.append(shift[2])

    # FITTING (linear)
    # y = p x
    def fit_simple(x, p):
        return p * x
    # and forcing the first point to be 0
    # (basically going from small float to int)
    shifts_plane[0] = 0
    shifts_row[0] = 0
    shifts_col[0] = 0

    # STRONG LINEARITY ASSUMPTION
    strong_linear_plane = shifts_plane[-1] / len(shifts_plane)
    strong_linear_row = shifts_row[-1] / len(shifts_plane)
    strong_linear_col = shifts_col[-1] / len(shifts_plane)

    x_plane = np.arange(0, len(shifts_plane), 1)
    x_row = np.arange(0, len(shifts_row), 1)
    x_col = np.arange(0, len(shifts_col), 1)

    def fit_simple(x, p):
        return p * x

    fit_plane_0 = curve_fit(
        fit_simple,
        x_plane,
        shifts_plane,
        p0=strong_linear_plane
        )
    fit_row_0 = curve_fit(
        fit_simple,
        x_row,
        shifts_row,
        p0=strong_linear_row
        )
    fit_col_0 = curve_fit(
        fit_simple,
        x_col,
        shifts_col,
        p0=strong_linear_col
        )

    # In the following plots, errorbars cover:
    #   +/- 1 z-step for the depth
    #   and +/- 1 pixel for x/y drift.

    summary = plt.figure('summary', figsize=(9, 7))

    summary.add_subplot(311)
    plt.title('Depth drift')
    plt.errorbar(x_plane, shifts_plane, yerr=1, fmt='o', color='dodgerblue')
    y_plane_0 = fit_plane_0[0] * x_plane
    plt.plot(x_plane, y_plane_0, '-', color='blue', alpha=.6)

    summary.add_subplot(312)
    plt.title('Vertical drift')
    plt.errorbar(x_row, shifts_row, yerr=1, fmt='o', color='orange')
    y_row_0 = fit_row_0[0] * x_row
    plt.plot(x_row, y_row_0, '-', color='red', alpha=.6)

    summary.add_subplot(313)
    plt.title('Horizontal drift')
    plt.errorbar(x_col, shifts_col, yerr=1, fmt='o', color='lightgreen')
    y_col_0 = fit_col_0[0] * x_col
    plt.plot(x_col, y_col_0, '-', color='green', alpha=.6)

    plt.tight_layout()
    plt.show()
    
    print(f'\nDepth drift: y = {fit_plane_0[0]} * x')   
    print(f'With only first and last points would give: {strong_linear_plane}\n')
    print(f'Vertical drift: y = {fit_row_0[0]} * x')   
    print(f'With only first and last points would give: {strong_linear_row}\n')
    print(f'Horizontal drift: y = {fit_col_0[0]} * x')   
    print(f'With only first and last points would give: {strong_linear_col}\n')


    # # ------------------------------------------------------------------------
    # # ------------------------------------------------------------------------

    # # Feature extraction
    # # TO merge images following Preibish et al. approximation to local entropy,
    # # two sigmas must be selected. The first one should be larger than the linear
    # # dimension of sensible features.
    # # e.g. feature = 10 pixels, value = 15 pixels (???)
    # # Find it with the cursos over the image,
    # # and then save it in "sigma" value.
    # #

    # feature_dimension= plt.figure('extract feature')


    #  # -----------------------------------------------------------------------
    # # ------------------------------------------------------------------------

    # # Saving all the hyperparameters in a .csv
    # # - cameras offset
    # # - drift fit parameters
    # # - sigma 
    with open(
        PARAMS_OUTPUT_FOLDER + '/hyperparameters.csv',
        'w',
        newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Camera vertical offset", 1])
        writer.writerow(["Camera horizontal offset", 1])
        writer.writerow(["T drift - depth", np.round(fit_plane_0[0][0], 2)])
        writer.writerow(["T drift - vertical", np.round(fit_row_0[0][0], 2)])
        writer.writerow(["T drift - horizontal", np.round(fit_col_0[0][0], 2)])
        
    # test read -> make a function!

    def read_hyperparameters(
        csv_file,
        shift_container,
        drift_container,
        soma_container):
        reader = list(csv.reader(csv_file, delimiter=','))
        shift_container[0] = reader[1][1]
        shift_container[1] = reader[2][1]
        # etc. etc.
        return None



