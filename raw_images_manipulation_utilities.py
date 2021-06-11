import os
import tifffile as tif
import math_utils as utils
import numpy as np
import matplotlib.pyplot as plt


def explore_cameras_offset(cam_1, cam_2, slices=[0], show=True):
    """Returns a tuple.
    """
    shift_row, shift_col = [], []
    for image in slices:
        cross = utils.spatial_xcorr_2D(cam_1[int(image), :, :],
                                       cam_2[int(image), :, :])
        row, col = utils.xcorr_peak_2D(cross)
        shift_row.append(-1 * row); shift_col.append(-1 * col)
    if(show):
        plt.figure('cameras offseet exploration')
        plt.plot(shift_row, 'b.-')
        plt.plot(shift_col, 'g.-')
        plt.xlabel('image')
        plt.ylabel('shift')
        plt.show()
    return (int(np.round(np.mean(shift_row))), 
            int(np.round(np.mean(shift_col))))

def cameras_shift_registration(views, shift):
    """ Cure the intrinsic shift between the two cameras by moving one set of 
        data by the shift found with a 2D cross-correlation.
        - views = volume  [slice_number, rows, cols]
        - shift =  tuple(rows, cols)
    """
    shifted = np.zeros((views.shape), dtype=np.uint16)
    for i in range(views.shape[0]):
        shifted[i, :, :] = utils.shift_image(views[i, : :], shift)
    return shifted

def merge_views(front, back, method='local_variance', sigma=10):
    """ Images are merged.
    """
    merged = np.zeros((front.shape), dtype=np.uint16)
    if method == 'average':
        # Super simple average: fast but poor quality
        merged[:, :, :] = np.int((front[:, :, :] + back[:, :, :] / 2))
    elif method == 'local_variance':
        # Images are merged using the approximation of local variance
        # similar to the one defined  in Preibish et al. 2008
        # 'Mosaicing of Single Plane Illumination Microscopy Images
        # Using Groupwise Registration and Fast Content-Based Image Fusion'
        # shorturl.at/nHMY5
        gaus_1 = utils.gaussian_2D(1024, sigma=sigma)
        Gaus_1 = utils.FT2(gaus_1)
        for i in range(front.shape[0]):
            Front = utils.FT2(front[i, :, :])
            Back = utils.FT2(back[i, :, :])
            front_weight = (utils.IFT2(Front - (Front * Gaus_1)))**2
            back_weigth = (utils.IFT2(Back - (Front * Gaus_1)))**2
            tot = front_weight + back_weigth
            merged[i, :, :] = np.real((front_weight * front[i, :, :] 
                                     + back_weigth * back[i, :, :])\
                                     / tot[:, :])
    return merged

def find_camera_registration_parameters(image_1, image_2):
    M, N = image_1.shape
    shift = [0, 0]
    image_1_n = utils.normalize_0_1(image_1)
    image_2_n = utils.normalize_0_1(image_2)
    cross = utils.spatial_Xcorr_2D(image_1_n, image_2_n)
    shift[0], shift[1] = np.unravel_index(np.argmax(cross), cross.shape) # it's ok even if pylint complains
    center = np.asarray([int(M/2), int(N/2)])
    shift[0] -= center[0]
    shift[1] -= center[1]
    return shift # row and col components (y, and x cartesian then)

def explore_camera_shifts(volumes_to_investigate,
                          slices_to_investigate,
                          RAW_SLICES_FOLDER,
                          VOLUME_SLICES,
                          show='y'):
    shifts = np.zeros((len(volumes_to_investigate), len(slices_to_investigate), 2))
    for iter_0, i in enumerate(volumes_to_investigate):
        vol_0 = utils.open_binary_volume_with_hotpixel_correction(RAW_SLICES_FOLDER + files_list[i][1], 
                                                            VOLUME_SLICES, IMAGES_DIMENSION)
        vol_1 = utils.open_binary_volume_with_hotpixel_correction(RAW_SLICES_FOLDER + files_list[i][2],
                                                            VOLUME_SLICES, IMAGES_DIMENSION)
        for iter_1, j in enumerate(slices_to_investigate):
            shifts[iter_0, iter_1, 0], shifts[iter_0, iter_1, 1] = \
                find_camera_registration_parameters(vol_0[j, :, :], vol_1[j, :, ::-1]) # nb the ::-1                                                                       
    if show=='y':
        print(shifts)
        print('\nMeans: ')
        print(round(np.mean(shifts[:, :, 0]), 2), ', ', round(np.mean(shifts[:, :, 1]),2))
        print('\nStds: ')
        print(round(np.std(shifts[:, :, 0]), 2), ', ', round(np.std(shifts[:, :, 1]), 2))
        print('\nN.B. Output has rounded int values.')
        figshift  = plt.figure('1')
        x = np.arange(0, len(np.mean((shifts[:, :, 0]),axis=0)), 1)
        figshift.add_subplot(211)
        plt.title('vertical')
        plt.errorbar(x, np.mean((shifts[:, :, 0]), axis=0), np.std((shifts[:, :, 0]), axis=0),fmt='o')
        plt.fill_between(x, np.mean((shifts[:, :, 0]), axis=0)-np.std((shifts[:, :, 0]), axis=0),\
                                 y2=np.mean((shifts[:, :, 0]), axis=0)+np.std((shifts[:, :, 0]), axis=0),\
                                    color='b',alpha=.3,interpolate=True)
        #plt.xlabel('selected volume [#]')
        plt.ylabel('shift [pixels]')
        plt.ylim(-30, -25)
        figshift.add_subplot(212)
        plt.title('horizontal')
        plt.errorbar(x, np.mean((shifts[:, :, 1]), axis=0), np.std((shifts[:, :, 1]), axis=0),fmt='ro')
        plt.fill_between(x, np.mean((shifts[:, :, 1]), axis=0)-np.std((shifts[:, :, 1]), axis=0),\
                                 y2=np.mean((shifts[:, :, 1]), axis=0)+np.std((shifts[:, :, 1]), axis=0),\
                                    color='r',alpha=.3,interpolate=True)
        plt.ylim(155, 161)
        plt.xlabel('selected slice [#]')
        plt.ylabel('shift [pixels]')
        figshift.tight_layout()
        plt.show()
        figshift.savefig('shifts_slices.png',transparent=True)

    return int(round(np.mean(shifts[:, :, 0]), 0)), int(round(np.mean(shifts[:, :, 1]), 0))
    


if __name__ == '__main__':
        
    # SETing UP

    VOLUME_SLICES = 10 # slices for each volume
    TOTAL_VOLUMES =  2 # raw volumes  (they have 2X the nof images)
    TOTAL_IMAGES = TOTAL_VOLUMES * VOLUME_SLICES # total of raw data-images
    IMAGES_DIMENSION = 1024 # assumed square images. N.b. try to have 2^n pixels
    BACKGROUND_FOLDER = '/home/ngc/Documents/ls_test_data/' # add a "/" at the eend
    RAW_SLICES_FOLDER = '/home/ngc/Documents/ls_test_data/' # or os.getcwd()
    VOLUMES_OUTPUT_FOLDER = '/home/ngc/Documents/ls_test_data/' # define the name of the subfolder for saving

    if os.path.isdir(VOLUMES_OUTPUT_FOLDER) == False:
        os.mkdir(VOLUMES_OUTPUT_FOLDER)

    # BACKGROUND_CAM_1 = tif.imread(BACKGROUND_FOLDER + 'Background_0.tif')
    # BACKGROUND_CAM_2 = tif.imread(BACKGROUND_FOLDER + 'Background_1.tif') 
    # volume = np.zeros((VOLUME_SLICES,
    #                    IMAGES_DIMENSION,
    #                    IMAGES_DIMENSION), dtype=np.uint16) # n.b. single volume
    
    # mean_background_1 = int(round(np.mean((BACKGROUND_CAM_1)), 0))
    # mean_background_2 = int(round(np.mean((BACKGROUND_CAM_2)), 0))
    # mean_background_1 = 0
    # mean_background_2 = 0 # background subtraction crea problemi... why?
    files_list = utils.files_names_list(TOTAL_VOLUMES)
    
    print(files_list) 

    # # CAMERAs mismatch COMPENNSATION

    volumes_to_investigate = np.arange(90, 100, 5)
    slices_to_investigate = np.arange(25, 30, 1)#np.asarray((3, 38, 40, 42))
    shifts = explore_camera_shifts(volumes_to_investigate, slices_to_investigate, RAW_SLICES_FOLDER, VOLUME_SLICES,
                                show='n')

