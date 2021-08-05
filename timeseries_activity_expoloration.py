import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys
import pickle
from scipy.optimize import curve_fit
# local folder with homemade libraries
sys.path.append('/home/ngc/Documents/GitHub/multiview_utilities')
# HOMEMADE LIBRARIES AND IMPORTS
import math_utils as matut
import raw_images_manipulation_utilities as rim
importlib.reload(matut);
importlib.reload(rim); # reload the modules for updating to eventual change


def extract_timepoint(
    data_list,
	timepoint, # every timepoint, a volume is selected
	save_path,
    IMAGES_DIMENSION=1024,
    RAW_SLICES_FOLDER='/home/ngc/Data',
    VOLUMES_OUTPUT_FOLDER = '/home/ngc/Data/data/output_for_params'
	):
    for i in range volumes:
        stack = rim.open_binary_stack(
            RAW_SLICES_FOLDER + data_list[i][1],
            BACKGROUND_FOLDER + '/Background_0.tif',
            size_x=IMAGES_DIMENSION,
            size_y=IMAGES_DIMENSION)

if __name__ == '__main__':

    #Local variables and constants.
    # Change file location and stack properties 
    VOLUME_SLICES = 161 # number of images for each volume
    TIMEPOINTS =  10 # aka number of volumes
    TOTAL_IMAGES = TIMEPOINTS * VOLUME_SLICES # total of raw data-images
    IMAGES_DIMENSION = 1024 # assumed square images. N.b. try to have 2^n pixels
    RAW_SLICES_FOLDER = '/home/ngc/Data/data'
    BACKGROUND_FOLDER = '/home/ngc/Data/data'
    VOLUMES_OUTPUT_FOLDER = '/home/ngc/Data/data/output_for_params'
    PARAMS_OUTPUT_FOLDER = '/home/ngc/Data/data'
    # create output folders if not present:
    if os.path.isdir(VOLUMES_OUTPUT_FOLDER) == False:
        os.mkdir(VOLUMES_OUTPUT_FOLDER)
    if os.path.isdir(PARAMS_OUTPUT_FOLDER) == False:
        os.mkdir(PARAMS_OUTPUT_FOLDER)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # list of names of all the volumes, defined as :
    # [timepoint (int), vol_cam_0 [str], vol_cam_1 (str)]
    # Next function uses default file names as saved by Labview software
    # (check the arguments in "rim" library for more details)
    data_list = rim.file_names_list(TIMEPOINTS)
    # Dictionary to group the hyperparameters calculated by this program:
    stack_0 = rim.open_binary_stack(
        RAW_SLICES_FOLDER + data_list[timepoint_for_offset_calculation][1],
        BACKGROUND_FOLDER + '/Background_0.tif',
        size_x=IMAGES_DIMENSION,
        size_y=IMAGES_DIMENSION)

