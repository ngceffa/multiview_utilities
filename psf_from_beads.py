"""
Code adapted from D.Nguyen 2021:
    https://github.com/Omnistic/MicrobeadsToPSF.git
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib
import pickle
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
# local folder with homemade libraries
import sys
sys.path.append('/home/ngc/Documents/GitHub/multiview_utilities')
# HOMEMADE LIBRARIES AND IMPORTS
import math_utils as matut
import raw_images_manipulation_utilities as rim
importlib.reload(matut); # 
importlib.reload(rim); # reload the modules for updating to eventual changes


if __name__=='__main__':

    # --------------------------------------------------------------------------
    #Local variables and constants.
    # Carefully read and change the variables in the next section
    # to encompass the data to be analysed.
    VOLUME_SLICES = 41 # number of images for each volume
    TIMEPOINTS =  1
    TOTAL_IMAGES = TIMEPOINTS * VOLUME_SLICES # total of raw data-images
    IMAGES_DIMENSION = 2304 # assumed square images. N.b. try to have 2^n pixels
    RAW_SLICES_FOLDER = '/home/ngc/Desktop/test_data/Beads20X_LowerC/beads'
    BACKGROUND_FOLDER = '/home/ngc/Desktop/test_data/Beads20X_LowerC/beads'

    z_width, x_width, y_width = [], [], []
    # (check arguments in "rim" library)
    data_list = rim.file_names_list(TIMEPOINTS)
    stack_0 = rim.open_binary_stack(
        RAW_SLICES_FOLDER + data_list[0][1],
        BACKGROUND_FOLDER + '/Background_0.tif',
        size_x=IMAGES_DIMENSION,
        size_y=IMAGES_DIMENSION)
    stack_1 = rim.open_binary_stack(
        RAW_SLICES_FOLDER + data_list[0][2],
        BACKGROUND_FOLDER + '/Background_1.tif',
        size_x=IMAGES_DIMENSION,
        size_y=IMAGES_DIMENSION)

    box = np.asarray((21, 21, 21))

    # x_val = np.arange(- int(box[0]/2), int(box[0]/2), 1)
    # y_val = np.arange(- int(box[1]/2), int(box[1]/2), 1)
    # z_val = np.arange(- int(box[2]/2), int(box[2]/2), 1)
    x_val = np.arange(0, box[0], 1)
    y_val = np.arange(0, box[1], 1)
    z_val = np.arange(0, box[2], 1)
    z_grid, x_grid, y_grid = np.meshgrid(z_val, x_val, y_val, indexing='ij')
    xdata = np.vstack((z_grid.ravel(), x_grid.ravel(), y_grid.ravel()))
    mean_b_z, mean_b_x, mean_b_y = 0, 0, 0
    
    beads = 0
    max_beads = 30
    z_chosen = 21
    print('\nNUMBER OF BEADS FOUND:\n')
    while beads < max_beads:

        x_max, y_max = np.unravel_index(
            np.argmax(stack_1[20, :, :]),
            stack_1[20, :, :].shape)

        if (
            x_max - box[1] / 2 > 0
            and x_max + box[1] / 2 < stack_0.shape[1]
            and y_max - box[2] / 2 > 0
            and y_max + box[2] / 2 < stack_0.shape[2]):
            substack = stack_1[
                int(z_chosen - box[0] / 2):int(z_chosen + box[0] / 2),
                int(x_max - box[1] / 2):int(x_max + box[1] / 2),
                int(y_max - box[2] / 2):int(y_max + box[2] / 2),
            ]

            bg = 1
            A_coeff = 1000
            x_0 = box[1] / 2
            y_0 = box[2] / 2
            z_0 = box[0] / 2
            x_sig = 3
            y_sig = x_sig
            z_sig = 4
            p0 = [bg, A_coeff, z_0, x_0, y_0, z_sig, x_sig, y_sig]
            # params = []
            popt, pcov = curve_fit(
                matut.gaus_3D_for_fit_2,
                xdata,
                substack.ravel(), 
                p0)
            fit = matut.gaus_3D_for_fit_2(xdata, *popt)
            fit = np.reshape(fit, (box[0], box[1], box[2]))
            z_width.append(np.abs(popt[5]))
            x_width.append(np.abs(popt[6]))
            y_width.append(np.abs(popt[7]))

            mean_b_z += np.abs(popt[5]) / max_beads
            mean_b_x += np.abs(popt[6]) / max_beads
            mean_b_y += np.abs(popt[7]) / max_beads
            stack_1[
                int(z_chosen - box[0] / 2):int(z_chosen + box[0] / 2),
                int(x_max - box[1] / 2):int(x_max + box[1] / 2),
                int(y_max - box[2] / 2):int(y_max + box[2] / 2),
            ] = 0
            beads += 1
        else:
            stack_1[
                int(z_chosen - box[0] / 2):int(z_chosen + box[0] / 2),
                int(x_max - box[1] / 2):int(x_max + box[1] / 2),
                int(y_max - box[2] / 2):int(y_max + box[2] / 2),
            ] = 0
        
        print(beads, end=', ', flush=True)
    

    pixel_size = .3 #um
    z_step = 1 #um
    print(
        '\nMean_b_z: ', np.round(mean_b_z * z_step, 2), 
        '\nMean_b_x' , np.round(mean_b_x * pixel_size, 2),
        '\nMean_b_y' , np.round(mean_b_y * pixel_size, 2))
    plt.figure('widths', figsize=(12, 8))
    plt.plot(z_width, 'o-', color='mediumorchid', lw=2, ms=7, alpha=.8, label='z')
    z_const = mean_b_z * np.ones((len(z_width)))
    plt.plot(z_const, '--', color='indigo', lw=2)
    plt.plot(x_width, 's-', color='indianred', lw=2, ms=7, alpha=.8, label='x')
    x_const = mean_b_x * np.ones((len(x_width)))
    plt.plot(x_const, '--', color='red', lw=2)
    plt.plot(y_width, '^-', color='dodgerblue', lw=2, ms=7, alpha=.8, label='y')
    y_const = mean_b_y * np.ones((len(y_width)))
    plt.plot(y_const, '--', color='blue', lw=2)
    plt.legend()
    plt.grid()
    plt.show()
    
    params_dict = {}

    # add "eliminate slides" in Jupyter


    # print(np.average(y_width[10:]) * pixel_size)
    # print(np.average(x_width) * pixel_size)
    # print(np.average(z_width) * z_step)

    # params_dict['bz': mean_b_z]

    # with open(RAW_SLICES_FOLDER + '/psf_fit_params.pkl', 'wb') as params_file:
    #     pickle.dump(params_dict, params_file, pickle.HIGHEST_PROTOCOL)

