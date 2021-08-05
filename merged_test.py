from lumped import merge_views
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import importlib
import os
import sys
import pickle
import napari as nap
from scipy.optimize import curve_fit
# local folder with homemade libraries
sys.path.append('/home/ngc/Documents/GitHub/multiview_utilities')
# HOMEMADE LIBRARIES AND IMPORTS
import math_utils as matut
import raw_images_manipulation_utilities as rim
importlib.reload(matut);
importlib.reload(rim); # reload the modules for updating to eventual changes

def deconvolve(image, psf, iterations=5):
    # object refers to the estimated "true" sample
    object = np.copy(image).astype(complex)
    # lucy-richardson in the for loop
    for k in range (iterations):
        step_0 = image / (matut.IFT2(matut.FT2(object) * matut.FT2(psf)))
        step_1 = matut.IFT2(matut.FT2(step_0) * np.conj(matut.FT2(psf)))
        object *= (step_1)**2
    return np.real(object)

def deconvolve_3D(image, psf, iterations=5):
    # object refers to the estimated "true" sample
    image = image.astype(complex)
    # lucy-richardson in the for loop
    for k in range (iterations):
        step_0 = image / (matut.IFT3(matut.FT3(image) * psf))
        print('k')
        step_1 = matut.IFT3(matut.FT3(step_0) * np.conj(psf))
        image *= (step_1)**2
        print(k)
    return np.real(image)

def psf_gauss_2D(dimension, sigmas):
    x, y, = np.meshgrid(np.arange(0, dimension, 1), 
                        np.arange(0, dimension, 1))
    return np.exp(- np.pi * ( ((x - dimension / 2.)**2 / (sigmas[0]**2)) 
                    + ((y - dimension / 2.)**2 / (sigmas[1]**2))))


def psf_gauss_3D(dimension, z_dim, sigmas):
    z, x, y = np.meshgrid(np.arange(0, z_dim, 1), 
                        np.arange(0, dimension, 1),
                        np.arange(0, dimension, 1))
    return np.exp(- np.pi * ( ((x - dimension / 2.)**2 / (sigmas[0]**2)) 
                    + ((y - dimension / 2.)**2 / (sigmas[1]**2))
                    + ((z - z_dim / 2.)**2 / (sigmas[2]**2))))

if __name__=='__main__':

    # --------------------------------------------------------------------------
    #Local variables and constants.
    # Carefully read and change the variables in the next section
    # to encompass the data to be analysed.
    VOLUME_SLICES = 161 # number of images for each volume
    TIMEPOINTS =  1
    TOTAL_IMAGES = TIMEPOINTS * VOLUME_SLICES # total of raw data-images
    IMAGES_DIMENSION = 1024 # assumed square images. N.b. try to have 2^n pixels
    RAW_SLICES_FOLDER = '/home/ngc/Data/data'
    BACKGROUND_FOLDER = '/home/ngc/Data/data'
    VOLUMES_OUTPUT_FOLDER = '/home/ngc/Data/data/one_sigma_values_comparison'
    PARAMS_OUTPUT_FOLDER = '/home/ngc/Data/data/output_for_params'
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
    with open(RAW_SLICES_FOLDER + '/hypeparameters.pkl', 'rb') as params:
        hyperparameters = pickle.load(params)
        shift = hyperparameters['shift']

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
        stack_0 = rim.open_binary_stack(
            RAW_SLICES_FOLDER + data_list[0][1],
            BACKGROUND_FOLDER + '/Background_0.tif',
            background_estimation_method='none',
            size_x=IMAGES_DIMENSION,
            size_y=IMAGES_DIMENSION)
        stack_1 = rim.open_binary_stack(
            RAW_SLICES_FOLDER + data_list[0][2],
            BACKGROUND_FOLDER + '/Background_1.tif',
            background_estimation_method='none',
            size_x=IMAGES_DIMENSION,
            size_y=IMAGES_DIMENSION)
        # stack_0 = rim.open_binary_stack(
        #     RAW_SLICES_FOLDER + '/dc_0.stack',
        #     BACKGROUND_FOLDER + '/Background_0.tif',
        #     size_x=IMAGES_DIMENSION,
        #     size_y=IMAGES_DIMENSION)
        # stack_1 = rim.open_binary_stack(
        #     RAW_SLICES_FOLDER + '/dc_1.stack',
        #     BACKGROUND_FOLDER + '/Background_1.tif',
        #     size_x=IMAGES_DIMENSION,
        #     size_y=IMAGES_DIMENSION)
        
        # plt.plot(stack_0[:, 450, 512])
        # plt.plot(stack_1[:, 450, 512])
        
        
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    for i in range(stack_1.shape[0]):
        stack_1[i, :, :] = matut.shift_image(stack_1[i, :, ::-1], shift)
    
    for i in range(0, 1, 1):
        merged = rim.merge_views(
            stack_0[45:60, 250:750, 250:750],
            stack_1[45:60, 250:750, 250:750],
            method='scigrad',
            sigma_1=1+i)
        merged[merged > 65000] = 0 # to be extra sure it does not exceed the limit
        merged.tofile(VOLUMES_OUTPUT_FOLDER +'/_'+str(i+1)+'_'+
            'scigrad_d10.stack')
    
    # stds = []
    # means = []
    # maxs = []
    # mins = []
    # for i in range(0, 40, 5):
    #     stack = np.fromfile(VOLUMES_OUTPUT_FOLDER +'/_'+str(i)+'_'+
    #          'from_raw.stack', dtype=np.uint16)
    #     size_z = int(stack.size / 1024 / 1024)
    #     stack = np.reshape(stack, (size_z, 1024, 1024))

    #     stds.append(np.std(stack))
    #     means.append(np.mean(stack))
    #     maxs.append(np.amax(stack))
    #     mins.append(np.amin(stack))

    # fig = plt.figure('compare')
    # fig.add_subplot(221)
    # plt.title('std')
    # plt.plot(stds, 'bo--')
    # fig.add_subplot(222)
    # plt.title('mean')
    # plt.plot(means, 'ro--')
    # fig.add_subplot(223)
    # plt.title('maxs')
    # plt.plot(maxs, 'ko--')
    # fig.add_subplot(224)
    # plt.title('mins')
    # plt.plot(mins, 'go--')
    # plt.show()

    
    # dec_then_merge = rim.open_binary_stack(
    #         VOLUMES_OUTPUT_FOLDER + \
    #             '/3d_dec_then_more_deriv_is_pow2_sigma_is_15.stack',
    #         BACKGROUND_FOLDER + '/Background_0.tif',
    #         size_x=IMAGES_DIMENSION,
    #         size_y=IMAGES_DIMENSION)
    # dec_2d_then_merge = rim.open_binary_stack(
    #         VOLUMES_OUTPUT_FOLDER + \
    #             '/after_dec_more_deriv_is_pow2_sigma_is_15.stack',
    #         BACKGROUND_FOLDER + '/Background_0.tif',
    #         size_x=IMAGES_DIMENSION,
    #         size_y=IMAGES_DIMENSION)

    # viewer = nap.Viewer()
    # new_layer = viewer.add_image(dec_then_merge, scale=(3, 1, 1))
    # new_layer_2 = viewer.add_image(dec_2d_then_merge, scale=(3, 1, 1))
    # nap.run()


    # sigmas = [3, 4]
    # psf = psf_gauss_2D(1024, sigmas)
    # # psf = matut.FT3(psf)

    # #raw_deconv_0 = np.zeros((stack_0.shape))
    # raw_deconv_1 = np.zeros((stack_1.shape))
    # for i in range(stack_0.shape[0]):
    #     #raw_deconv_0[i, :, :] = deconvolve(stack_0[i, :, :], psf, 3)
    #     raw_deconv_1[i, :, :] = deconvolve(stack_1[i, :, :], psf, 3)
    # #raw_deconv_0 = deconvolve_3D(stack_0, psf, 3)
    # # raw_deconv_1 = deconvolve_3D(stack_1, psf, 3)
    # #raw_deconv_0[raw_deconv_0 > 65000] = 0
    # raw_deconv_1[raw_deconv_1 > 65000] = 0

    # #raw_deconv_0 = raw_deconv_0.astype(np.uint16)
    # raw_deconv_1 = raw_deconv_1.astype(np.uint16)

    # #raw_deconv_0.tofile(RAW_SLICES_FOLDER +
    #     #"/raw_dec_3_0.stack")
    # raw_deconv_1.tofile(RAW_SLICES_FOLDER +
    #     "/raw_dec_3_1.stack")