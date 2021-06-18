import numpy as np
from numpy.lib.function_base import meshgrid
import tifffile as tif
import multiprocessing as mp
import os
from functools import partial # to use map with second fixed argument="keyword"
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift, ifftshift, fftfreq
import matplotlib.pyplot as plt
import pywt
import time
import PIL
from pprint import pprint
#from skimage.tranforms import resize
#import mayavi.mlab as mlab

font = {'size'   : 18}

plt.rc('font', **font)

def deconvolve(image, psf, iterations=5):
    # object refers to the estimated "true" sample
    object = np.copy(image).astype(complex)
    # lucy-richardson in the for loop
    for k in range (iterations):
        step_0 = image / (IFT2(FT2(object) * FT2(psf)))
        step_1 = IFT2(FT2(step_0) * np.conj(FT2(psf)))
        object *= step_1
    return np.real(object)

def temporal_shift_exploration(vol):
    return None

def temporal_shift_correction(vol):
    return None

def create_volumes_time_series(files_list, TOTAL_VOLUMES, VOLUME_SLICES, 
                                IMAGES_DIMENSION, VOLUMES_OUTPUT_FOLDER):
    time_series = np.zeros((TOTAL_VOLUMES,
                            VOLUME_SLICES,
                            IMAGES_DIMENSION,
                            IMAGES_DIMENSION), dtype=np.uint16)
    for i in range(TOTAL_VOLUMES):
        time_series[i] = open_binary_volume_with_hotpixel_correction(VOLUMES_OUTPUT_FOLDER + 'volume_'
                                                                    +files_list[i][0]+'.stack', 
                                                                    VOLUME_SLICES, IMAGES_DIMENSION)
    time_series.tofile(VOLUMES_OUTPUT_FOLDER+"time_series.stack")
    return None

# yes
def files_names_list(total_volumes, seed_0='/SPC00_TM', 
                     seed_1='_ANG000_CM', seed_2='_CHN00_PH0'):
    files_list = []
    j = 0
    for i in range(total_volumes):
        temp_list = [str(i)]
        for k in range(0, 2):
            temp_list.append(seed_0 + f'{j:05}' + seed_1
                            + str(k) + seed_2 + ".stack")
        files_list.append(temp_list)
        j += 1
    return files_list

def write_binned_display(name, volume,  M, N, reduction_factor = 8):
    dim = volume.shape[1]
    new_dim = int(dim / reduction_factor)
    result = np.zeros((int(M * new_dim), int(N * new_dim)), dtype=np.uint16)
    k = 0
    for i in range(M):
        for j in range(N):
            if k < volume.shape[0]:
                result[i * new_dim:i * new_dim + new_dim, j * new_dim:j * new_dim + new_dim] = volume[k, ::4, ::4]
            k += 1
    tif.imwrite(name, result, dtype=np.uint16)
    return None

# yes
def normalize_0_1(array):
    maximum, minimum = np.amax(array), np.amin(array)
    normalized = np.zeros(array.shape)
    delta = maximum - minimum
    normalized = (array - minimum) / delta
    return normalized

# yes
def spatial_Xcorr_2D(f, g, pad=False):
    """
    Cross-correlation between two 2D functions: (f**g).
    """
    if pad==True:
        M, N = f.shape[0], f.shape[1]
        one, two = np.pad(np.copy(f),
                        ((int(M/2), int(M/2)),
                        (int(N/2), int(N/2))),
                        mode = 'constant',
                        constant_values=(0,0)),\
                np.pad(np.copy(g),
                        ((int(M/2), int(M/2)),
                        (int(N/2), int(N/2))),
                        mode = 'constant', 
                        constant_values=(0,0))                  
        ONE, TWO =   FT2(one), FT2(two)
        spatial_cross = ifftshift(ifft2(ifftshift(ONE) * np.conj(ifftshift(TWO))))\
                        [int(M/2) :int(M/2+M), int(N/2) : int(N/2+N)]
    else:
        M, N = f.shape[0], f.shape[1]                 
        ONE, TWO =   FT2(f), FT2(g)
        spatial_cross = ifftshift(
            ifft2(ifftshift(ONE) * np.conj(ifftshift(TWO))))
    spatial_cross = normalize_0_1(spatial_cross)
    return np.abs(spatial_cross)

# yes
def FT2(f):
    return (fftshift(fft2(ifftshift(f))))

#yes
def IFT2(f):
    return (fftshift(ifft2(ifftshift(f))))

#yes
def find_camera_registration_parameters(image_1, image_2):
    M, N = image_1.shape
    shift = [0, 0]
    image_1_n = normalize_0_1(image_1)
    image_2_n = normalize_0_1(image_2)
    cross = spatial_Xcorr_2D(image_1_n, image_2_n)
    shift[0], shift[1] = np.unravel_index(
        np.argmax(cross),
        cross.shape) # it's ok even if pylint complains
    center = np.asarray([int(M/2), int(N/2)])
    shift[0] -= center[0]
    shift[1] -= center[1]
    return shift # row and col components (y, and x cartesian then)

# 2D images
def shift_image(image, shift):
    H, W = image.shape # Z = num. of images in stack; M, N = rows, cols;
    shifted = np.zeros((H, W),dtype = np.uint16)
    if shift[0] > 0: shifted[int(shift[0]):, :] = image[:int(H - shift[0]), :] # shift up
    elif shift[0] < 0: shifted[:int(H - shift[0]), :] = image[int(shift[0]):, :] # shift down
    if shift[1] < 0: shifted[:, :int(W - shift[1])] = shifted[:, int(shift[1]):] # shift left
    elif shift[1] > 0: shifted[:, int(shift[1]):] = shifted[:, :int(W - shift[1])] # shift right
    return shifted

# 3D stacks
def shift_views(views, shift):
    slices, H, W = views.shape # Z = num. of images in stack; M, N = rows, cols;
    shifted = np.copy(views)
    temp_shifted = np.copy(views)
    # aggiungi il riempimento dello spazio lasciato libero dallo shift.
    if shift[0] >= 0: temp_shifted[:, int(shift[0]):, :] = views[:, :int(H - shift[0]), :] # shift up
    elif shift[0] < 0: temp_shifted[:, :int(H + shift[0]), :] = views[:, int(-1 * shift[0]):, :] # shift down
    if shift[1] < 0: shifted[:, :, :int(W + shift[1])] = temp_shifted[:, :, int(-1 * shift[1]):] # shift left
    elif shift[1] >= 0: shifted[:, :, int(shift[1]):] = temp_shifted[:, :, :int(W - shift[1])] # shift right
    return shifted

def rescale_to_8bit(array, old_max=2000):
    array_8bit = np.zeros((array.shape), dtype=np.uint8)
    minimum = np.amin(array)
    amplitude = (255. / (old_max - minimum))
    array_8bit[:, :] = (array[:, :] - minimum) * amplitude
    return array_8bit

def stretch_range_in_16_bits(array, wiggle_max=2000):
    array_rescaled = np.zeros((array.shape), dtype=np.uint16)
    old_max = np.amax(array) + wiggle_max
    minimum = np.amin(array)
    amplitude = (2**16. / (old_max - minimum))
    array_rescaled[:, :] = (array[:, :] - minimum) * amplitude
    return array_rescaled

def numbered_file_list(total_volumes, volume_slices):
    files_list = []
    for i in range(total_volumes):
        temp_list = []
        for j in range(volume_slices * i, volume_slices * i + volume_slices, 1):
            temp_list.append(str(j)+".tif")
        files_list.append(temp_list)
    return files_list

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" %path)
    return None

def organize_volumes(TOTAL_VOLUMES,
                     VOLUME_SLICES,
                     IMAGES_DIMENSION,
                     INPUT_FOLDER,
                     VOLUMES_OUTPUT_FOLDER):
    """Re-saves the raw slices into single volumes.

    Parameters
    ----------
    TOTAL_VOLUMES : int
        Total number of volumes collected.
    VOLUME_SLICES : int
        Number of slices for each volume. N.B. this considers both the cameras views.
    VOLUMES_OUTPUT_FOLDER : str
        Where to save.
    IMAGES_DIMENSION: int
        Dimension of single images (supposed square).

    Returns
    -------
    None
    """
    temp_images = np.zeros((VOLUME_SLICES, IMAGES_DIMENSION, IMAGES_DIMENSION), dtype=np.uint16)
    for i in range(TOTAL_VOLUMES):
        for j in range(VOLUME_SLICES * i, VOLUME_SLICES * i + VOLUME_SLICES, 1):
            temp_images[j, :, :] = tif.imread(INPUT_FOLDER + str(j)+'.tif')
        tif.imwrite(VOLUMES_OUTPUT_FOLDER+"/volume_"+str(i)+".tif", temp_images)
    return None

def organize_volumes_8_bit(TOTAL_VOLUMES,
                           VOLUME_SLICES,
                           IMAGES_DIMENSION,
                           INPUT_FOLDER,
                           VOLUMES_OUTPUT_FOLDER):
    temp_images = np.zeros((VOLUME_SLICES, IMAGES_DIMENSION, IMAGES_DIMENSION), dtype=np.uint16)
    temp_images_8_bit = np.zeros((VOLUME_SLICES, IMAGES_DIMENSION, IMAGES_DIMENSION), dtype=np.uint8)
    for i in range(TOTAL_VOLUMES):
        for j in range(VOLUME_SLICES * i, VOLUME_SLICES * i + VOLUME_SLICES, 1):
            temp_images[j, :, :] = tif.imread(INPUT_FOLDER + str(j)+'.tif')
            temp_images_8_bit[j, :, :] = rescale_to_8bit(temp_images[j, :, :])
        tif.imwrite(VOLUMES_OUTPUT_FOLDER+"/volume_8_bit_"+str(i)+".tif",
                    temp_images_8_bit,
                    dtype=np.uint8)
    return None

def merge_views(
    front,
    back,
    method='mean',
    front_deconv=None, # deconvolved versions
    back_deconv=None
    ):
    # Using "front" and "back" camera views
    merged = np.zeros((front.shape), dtype=np.uint16)

    if method == 'mean':
        merged[:, :, :] = (front[:, :, :] + back[:, :, :]) / 2

    elif method == 'max':
        merged[:, :, :] = np.maximum(front[:, :, :], back[:, :, :])

    elif method == 'linear_weights':
        weights =  np.linspace(0, 1, front.shape[0])
        for i in range(front.shape[0]):
            merged[i, :, :] =  front[i, :, :] * weights[i] \
                            + back[i, :, :] * weights[weights.shape[0]-i-1]
    
    elif method == 'var_weights':
        std_front = np.zeros((front.shape[0]), dtype=np.single)
        std_back = np.zeros((front.shape[0]), dtype=np.single)
        std_front[:] = np.var(front, axis=(1, 2))
        std_back[:] = np.var(back, axis=(1, 2))
        std_ratio = std_front / std_back
        old_max = np.amax(std_ratio)
        minimum = np.amin(std_ratio)
        amplitude = (1. / (old_max - minimum))
        std_rescaled_0_1 = np.zeros((std_ratio.shape), dtype=np.single)
        std_rescaled_0_1 = (std_ratio - minimum) * amplitude
        for i in range(front.shape[0]):
            merged[i, :, :] =  front[i, :, :] * std_rescaled_0_1[i]\
                               + back[i, :, :] * (1 - std_rescaled_0_1[i])
    
    elif method == 'preib':

        # Compute the two gaussian filters
        sigma_1 = 42
        sigma_2 = 88
        x = np.arange(0, merged.shape[1], 1)
        y = np.arange(0, merged.shape[2], 1)
        x, y = np,meshgrid(x, y)
        gaus_1 = np.exp(-x**2 / (2 * sigma_1**2)) \
                + np.exp(-y**2 / (2 * sigma_1**2))
        gaus_2 = np.exp(-x**2 / (2 * sigma_2**2)) \
                + np.exp(-y**2 / (2 * sigma_2**2))
        gaus_1_ft = FT2(gaus_1)
        gaus_2_ft = FT2(gaus_2)
        
        # Compute the weights for views summation.
        for i in range(front.shape[0]):
            front_ft = FT2(front[i, :, :])
            back_ft = FT2(back[i, :, :])
            front_weights = IFT2(front_ft * gaus_1_ft)
            front_weights = (front[i, :, :] - front_weights)**2
            front_weights = IFT2(gaus_2_ft * FT2(front_weights))
            back_ft = FT2(back[i, :, :])
            back_weights = IFT2(back_ft * gaus_1_ft)
            back_weights = (back[i, :, :] - back_weights)**2
            back_weights = IFT2(gaus_2_ft * FT2(back_weights))
        norm = front_weights + back_weights

        # Weighted sum.
        for i in range(front.shape[0]):
            merged[i, :, :] = (
                front_weights[i, :, :] * front[i, :, :]
                + back_weights[i, :, :] * back[i, :, :])\
                / (norm[i, :, :])

    elif method == 'use_deconvolved':
        front_weights = front_deconv - front
        back_weights = back_deconv - back
        norm = front_weights + back_weights
        for i in range(front.shape[0]):
            merged[i, :, :] = (
                (front_weights[i, :, :] * front[i, :, :]
                + back_weights[i, :, :] * back[i, :, :])
                / norm[i, :, :])
    return merged

# this will be mapped to multiprocessing
# do I want to return anything? Or should wwe simply save the volumes?
# (for large acquisitions could be crazy to keep eveerything opened)
def organize_volumes_from_list_2(volume_list):
    
    IMAGES_DIMENSION = tif.imread('0.tif').shape[0]
    volume_slices = len(volume_list)
    temp_images = np.zeros((volume_slices, IMAGES_DIMENSION, IMAGES_DIMENSION), dtype=np.uint16)
    j = 0
    for i in volume_list:
        temp_images[j, :, :] = tif.imread(i)
        # can do something to single images
        j += 1
    # can do something to the volume
    tif.imwrite("volume_"+str(i)+".tif", temp_images)
    return None

def mp_organize_volumes_from_list_2(function, slices_list):
    pool = mp.Pool(os.cpu_count()-1)
    results = pool.map(function, [slice for slice in slices_list]) #<-- nested list: it works! <-> >-<
    pool.close()
    return results

def organize_volumes_from_list(volume_list, whereare, background_1, background_2, shift):
    # gli sto passando una list di double-views
    IMAGES_DIMENSION = tif.imread(whereare + volume_list[0]).shape[0]
    volume_slices = len(volume_list)
    merged_volume_slices = int(volume_slices/2)
    temp_images = np.zeros((volume_slices, IMAGES_DIMENSION, IMAGES_DIMENSION), dtype=np.uint16)
    single_volume = np.zeros((merged_volume_slices, IMAGES_DIMENSION, IMAGES_DIMENSION), dtype=np.uint16)

    for i in range(merged_volume_slices):
        temp_images[i, :, :] = tif.imread(whereare+volume_list[i])
        temp_images[i, :, :] = temp_images[i, :, :]  - background_1
        temp_images[i+merged_volume_slices, :, :] = \
                            tif.imread(whereare+volume_list[i+merged_volume_slices]) #check the -1
        temp_images[i+merged_volume_slices, :, :] = \
                            temp_images[i+merged_volume_slices, :, ::-1]  - background_2
        temp_images[i+merged_volume_slices, :, :] = \
                            shift_image( temp_images[i+merged_volume_slices, :, :] , shift)
        single_volume[i] = merge_views(temp_images[i, : , :], temp_images[i + merged_volume_slices, : ,:],\
                                        method='max')

    # for i in range(merged_volume_slices:
    #     single_volume[i] = merge_views(temp_images[i, : , :], temp_images[i + merged_volume_slices, : ,:])


        # can do something to the volume
    #tif.imwrite("volume_"+str(i)+".tif", temp_images)
    return single_volume # if I return None is quite faster

def numbered_file_list_to_save(total_volumes):
    files_list = []
    for i in range(total_volumes):
        temp_list = [str(i)]
        for j in range(2 * i, 2 * i + 2, 1):
            temp_list.append(str(j)+".stack")
        files_list.append(temp_list)
    return files_list

#yes
def organize_volumes_from_list_and_save(volume_list, whereare, volume_slices, images_dimension, \
                                        background_1, background_2, shift, save_folder):
    # gli sto passando una list di volume views
    temp_images_1 = open_binary_volume_with_hotpixel_correction(\
                                                whereare + volume_list[1], volume_slices, images_dimension)
    
    temp_images_1 = np.maximum(temp_images_1 - background_1, 0).astype(np.uint16)
    #temp_images_1_8_bit = stretch_range_in_16_bits(temp_images_1)

    temp_images_2 = open_binary_volume_with_hotpixel_correction(\
                                                whereare + volume_list[2], volume_slices, images_dimension)
    temp_images_2 = np.maximum(temp_images_2[:, :, ::-1] - background_2, 0).astype(np.uint16)
    #temp_images_2_8_bit = stretch_range_in_16_bits(temp_images_2)   
    #here! shift all the volume   
    temp_images_2 = shift_views(temp_images_2, shift)
    #here! merge all the volume
    single_volume = merge_views(temp_images_1,\
                                    temp_images_2,\
                                    method='sharpness_weights')

    # for i in range(merged_volume_slices:
    #     single_volume[i] = merge_views(temp_images[i, : , :], temp_images[i + merged_volume_slices, : ,:])


        # can do something to the volume
    #tif.imwrite("volume_"+volume_list[0]+".tif", single_volume)
    single_volume.tofile(save_folder+"volume_" + volume_list[0] + ".stack")
    #write_binned_display("volume_" + volume_list[0] + ".tif", single_volume,  9, 9, reduction_factor = 4)
    return None # if I return None is quite faster

#yes
def mp_organize_volumes_from_list(function, where, volume_slices, images_dimension, \
                                    background_1, background_2, shift, slices_list, save_folder, CPUs=None):
    if CPUs==None:
        pool = mp.Pool(int(os.cpu_count()/2))
    else:
        pool = mp.Pool(CPUs)
    results = pool.map(partial(function, whereare=where, volume_slices=volume_slices, images_dimension=images_dimension,\
                        background_1=background_1, background_2=background_2, shift=shift, save_folder=save_folder), \
                                         [slice for slice in slices_list])

    # first entry in slices_ist is the growing nuber
    # like list[[str(0), slices_vollume_0], [[str(1)][slices_vol_1],]]
    # etc etc
    # str(i) is used to write the correspondig volume --> this allows to keep track of the time.
    pool.close()
    return None

def temporal_registration(volume_couples): # ->  HOW MANY??? DEFINE A "delta time interval"
    # volume_couples = [['i', 'vol_1.tif', 'vol_2.tif']] -> can do parallel corrrelations and return a list of
    # shifts --> these can be then checked, plotted (assume linear, small displacement or no displacement at all...)
    # use 3D FFT...
    # find linear  relationshipe  before (like the merging?) from oneor 2 selected slices
    return None

def open_binary_volume(name, VOLUME_SLICES, IMAGES_DIMENSION, format=np.uint16):
    volume = open(name, 'rb')
    raw_array = np.fromfile(volume, dtype=format)
    volume_array =  np.reshape(raw_array, (VOLUME_SLICES, IMAGES_DIMENSION, IMAGES_DIMENSION))
    volume.close()
    return volume_array

def open_binary_volume_with_hotpixel_correction(name, VOLUME_SLICES, IMAGES_DIMENSION, format=np.uint16):
    """ It also performs hotpixel correction and background subtraction?"""
    volume = open(name, 'rb')
    raw_array = np.fromfile(volume, dtype=format)
    raw_array[raw_array[:] > 63000] = 0
    volume_array =  np.reshape(raw_array, (VOLUME_SLICES, IMAGES_DIMENSION, IMAGES_DIMENSION))
    volume.close()
    return volume_array

def open_binary_volume_series_with_hotpixel_correction(name,
                                                       time_steps,
                                                       VOLUME_SLICES,
                                                       IMAGES_DIMENSION,
                                                       format=np.uint16):
    """ It also performs hotpixel correction and background subtraction?"""
    volume = open(name, 'rb')
    raw_array = np.fromfile(volume, dtype=format)
    raw_array[raw_array[:] > 63000] = 0
    volume_array =  np.reshape(raw_array, (time_steps, VOLUME_SLICES, IMAGES_DIMENSION, IMAGES_DIMENSION))
    volume.close()
    return volume_array

# yes
def explore_camera_shifts(
    volumes_to_investigate,
    slices_to_investigate,
    stack_path,
    background_path,
    files_list,
    show='y'):
    num_volumes = len(volumes_to_investigate)
    num_slices = len(slices_to_investigate)
    shifts = np.zeros((num_volumes, num_slices, 2))
    for iter_0, i in enumerate(volumes_to_investigate):
        vol_0 = open_binary_stack(
            stack_path + files_list[i][1],
            background_path + '/Background_0.tif'
            )
        vol_1 = open_binary_stack(
            stack_path + files_list[i][2],
            background_path + '/Background_1.tif'
            )
        for iter_1, j in enumerate(slices_to_investigate):
            shifts[iter_0, iter_1, 0], shifts[iter_0, iter_1, 1] = \
                find_camera_registration_parameters(
                    vol_0[j, :, :],
                    vol_1[j, : ,::-1]) # inversion of second camera                                                                       
    shift_0 = np.mean(shifts[:, :, 0])
    shift_1 = np.mean(shifts[:, :, 1])
    std_0 = np.std(shifts[:, :, 0])
    std_1 = np.std(shifts[:, :, 1])
    if show=='y':
        print(shifts)
        print('\nMeans: ')
        print(round(shift_0, 0), ', ', round(shift_1, 0))
        print('\nStds: ')
        print(round(std_0, 2), ', ', round(std_1, 2))
        print('\nN.B. Output has rounded int values.')
        figshift  = plt.figure('1')
        x = np.arange(0, len(np.mean((shifts[:, :, 0]),axis=0)), 1)
        figshift.add_subplot(211)
        plt.title('vertical')
        plt.errorbar(
            x,
            np.mean((shifts[:, :, 0]), axis=0),
            np.std((shifts[:, :, 0]), axis=0),
            fmt='o'
            )
        plt.fill_between(
            x,
            np.mean((shifts[:, :, 0]), axis=0) - np.std((shifts[:, :, 0]), axis=0),
            y2=np.mean((shifts[:, :, 0]), axis=0) + np.std((shifts[:, :, 0]), axis=0),
            color='b',
            alpha=.3,
            interpolate=True)
        plt.ylabel('shift [pixels]')
        plt.ylim(-30, -25)
        figshift.add_subplot(212)
        plt.title('horizontal')
        plt.errorbar(
            x,
            np.mean((shifts[:, :, 1]), axis=0),
            np.std((shifts[:, :, 1]), axis=0),
            fmt='ro')
        plt.fill_between(
            x,
            np.mean((shifts[:, :, 1]), axis=0)-np.std((shifts[:, :, 1]), axis=0),
            y2=np.mean((shifts[:, :, 1]), axis=0)+np.std((shifts[:, :, 1]), axis=0),
            color='r',
            alpha=.3,
            interpolate=True)
        plt.ylim(155, 161)
        plt.xlabel('selected slice [#]')
        plt.ylabel('shift [pixels]')
        figshift.tight_layout()
        plt.show()
        figshift.savefig('shifts_slices.png',transparent=True)
    return int(round(shift_0, 0)), int(round(shift_1, 0))

def open_binary_stack(
    stack_path,
    background_path,
    background_estimation_method='max',
    use_int=True,
    size_x=2304,
    size_y=2304,
    file_type=np.uint16
    ):
    stack_original = np.fromfile(stack_path, dtype=file_type)
    # Determine Z size automatically based on the array size
    size_z = int(stack_original.size / size_x / size_y)
    # Reshape the stack based on known dimensions
    stack = np.reshape(stack_original, (size_z, size_y, size_x))
    type_max = np.iinfo(stack.dtype).max
    type_min = np.iinfo(stack.dtype).min
    # hotpixels correction
    stack[stack == type_max] = type_min
    # open background images and subtract a value based on the
    # background_estimation_method
    background = tif.imread(background_path)
    if background_estimation_method == 'max':
        background = np.amax(background)
    elif background_estimation_method == 'min':
        background = np.amix(background)
    elif background_estimation_method == 'mean':
        background = np.mean(background)
    else:
        print('wrong background evaluation method selected')
        return None
    stack_subtracted = stack.astype(np.float16) - background
    stack_subtracted[stack_subtracted[:, :, :] < 0] = 0
    if use_int == True: return stack_subtracted.astype(np.uint16)
    else: return stack_subtracted

if __name__ == '__main__':

    # SETing UP

    TOTAL_VOLUMES =  2 # raw volumes  (they have 2X the nof images)
    VOLUME_SLICES = 61 # slices for each volume
    TOTAL_IMAGES = TOTAL_VOLUMES * VOLUME_SLICES # total of raw data-images
    IMAGES_DIMENSION = 2304 # assumed square images. N.b. try to have 2^n pixels
    BACKGROUND_FOLDER = '/home/ngc/Data/20210614/20X_488_20210614_151940'
    RAW_SLICES_FOLDER = '/home/ngc/Data/20210614/20X_488_20210614_151940'
    VOLUMES_OUTPUT_FOLDER = '/home/ngc/Data/20210614/20X_488_20210614_151940'
    # define the name of the subfolder for saving

    if os.path.isdir(VOLUMES_OUTPUT_FOLDER) == False:
        os.mkdir(VOLUMES_OUTPUT_FOLDER)

    files_list = files_names_list(TOTAL_VOLUMES)

    single = open_binary_stack(
        RAW_SLICES_FOLDER + files_list[0][1],
        BACKGROUND_FOLDER + '/Background_0.tif'
        )

    # CAMERA SHIFT COMPENNSATION

    volumes_to_investigate = np.arange(0, 1, 1)
    slices_to_investigate = np.arange(20, 25, 1)
    shifts = explore_camera_shifts(
        volumes_to_investigate,
        slices_to_investigate,
        RAW_SLICES_FOLDER,
        RAW_SLICES_FOLDER,
        files_list,
        show='y')

    # #files_list = files_names_list(TOTAL_VOLUMES)
    # start = time.time()
    # mp_organize_volumes_from_list(organize_volumes_from_list_and_save,
    #                               RAW_SLICES_FOLDER,\
    #                               VOLUME_SLICES,
    #                               IMAGES_DIMENSION,
    #                               background_1,
    #                               background_2, 
    #                               shifts, files_list,
    #                               VOLUMES_OUTPUT_FOLDER)
    # print(time.time()-start)

    # # start = time.time()
    # create_volumes_time_series(files_list, 100, VOLUME_SLICES,
    #                             IMAGES_DIMENSION, VOLUMES_OUTPUT_FOLDER)
    # print(time.time()-start)

    ## temporal registration
    # find parameters (hp linear)

    # transform


    # import napari

    # # vol = open_binary_volume_series_with_hotpixel_correction(VOLUMES_OUTPUT_FOLDER + 'time_series.stack', 100, 
    # #                                                         31, IMAGES_DIMENSION)
    # vol = open_binary_volume_with_hotpixel_correction(VOLUMES_OUTPUT_FOLDER + 'volume_1.stack', 31, 
    #                                                         IMAGES_DIMENSION)
    # data = vol[25,:,:]

    # with napari.gui_qt():
    #     viewer = napari.view_image(vol, rgb=False, scale=[15, 1, 1])









#     view_1 = open_binary_volume(RAW_SLICES_FOLDER + '0.stack', VOLUME_SLICES, IMAGES_DIMENSION)
#     view_2 = open_binary_volume(RAW_SLICES_FOLDER + '1.stack', VOLUME_SLICES, IMAGES_DIMENSION)
#     comparison_slice = 40
#     shift = find_camera_registration_parameters(view_1[comparison_slice, :, :], view_2[comparison_slice, :, :])

#     # instead of MERGING VOLUMES, keep 2 stacks like twwo channels/colors 

#     start = time.time()
#     mp_organize_volumes_from_list(organize_volumes_from_list_and_save, RAW_SLICES_FOLDER,\
#                                     VOLUME_SLICES, IMAGES_DIMENSION, 50, 50, \
#                                   shift, files_list[1:], 60)
#     print(time.time()-start)

#     # # vol = open_binary_volume(RAW_SLICES_FOLDER+'3.stack', VOLUME_SLICES, IMAGES_DIMENSION, format=np.uint16)
#     # # # print(type(vol[0,0,0]))

#     # #tif.imwrite('test_sharp.tif', vol, dtype=np.uint16)

#     # check = np.zeros((39, 1024, 1024), dtype=np.uint32)
#     # check2 = np.zeros((39, 1024, 1024), dtype=np.uint32)

#     # for i in range(1, 40):
#     #     vol = open_binary_volume_with_hotpixel_correction(\
#     #                                     'volume_'+str(i)+'.stack', VOLUME_SLICES, IMAGES_DIMENSION, format=np.uint16)
#     #     print(np.amax(vol))
#     #     check[i-1, :, :] = (np.sum(vol[30:40, :, :], axis=(0)))**2
#     # check2 = stretch_range_in_16_bits(check, wiggle_max=0)
#     # plt.plot(np.sum(check2[:, 365:370, 620:630], axis=(1, 2))/\
#     #                 np.amax(np.sum(check2[:, 365:370, 620:630], axis=(1, 2))))
#     # plt.show()
#     # tif.imwrite('test_t.tif', check2, dtype=np.uint16)
    


#     # # ciao1 = tif.imread('test_weights_ratio.tif')
#     # # ciao2 = tif.imread('test_linear.tif')
#     # # plt.figure('test')
#     # # plt.plot(ciao1[0, 704,:], label='ratio')
#     # # plt.plot(ciao2[0, 704,:], '--', label='linear', alpha=.6, )
#     # # plt.legend()
#     # # plt.grid()
#     # # plt.show()
    
#     # # SOMETIMES THIGS APPEAR TO BE SAVED IN ( BITS FROM TIF)
#     # # MAYBE HOT PIXELS?
#     # # BUT IT SHOULD ALWAYS BE THE CASE ANYWAY; WITH ANY MERGE

#     # # vol0 = open_binary_volume(
#     # #             RAW_SLICES_FOLDER+'4.stack', VOLUME_SLICES, IMAGES_DIMENSION, format=np.uint16)
#     # # vol1 = open_binary_volume(
#     # #             RAW_SLICES_FOLDER+'5.stack', VOLUME_SLICES, IMAGES_DIMENSION, format=np.uint16)
    
#     # # x0 = np.std(vol0, axis=(1,2))
#     # # x1 = np.std(vol1, axis=(1,2))

#     # # # plt.figure('mnaxs')
#     # # # plt.plot(x0, 'b')
#     # # # plt.plot(x1, 'r')
#     # # # plt.plot((x0-x1>0)*250, 'k')
#     # # # plt.show()


#     # # plt.imshow(vol1[60, :, :])
#     # # plt.show()
#     # -*- coding: utf-8 -*-