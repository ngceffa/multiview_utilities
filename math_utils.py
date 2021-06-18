import numpy as np 
from numba import jit
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack as ft


def FT2(f):
    """ 2D Inverse Fourier Transform, with proper shift
    """
    return (ft.fftshift(ft.fft2(ft.ifftshift(f))))

def IFT2(f):
    """ 2D Fourier Transform, with proper shift
    """
    return (ft.fftshift(ft.ifft2(ft.ifftshift(f))))

def FT3(f):
    """ 3D Fourier Transform, with proper shift
    """
    return ft.fftshift(ft.fftn(ft.ifftshift(f)))

def IFT3(F):
    """ 3D Inverse Fourier Transform, with proper shift
    """
    return ft.ifftshift(ft.ifftn(ft.fftshift(F)))

def convert_to_16_bit(array):
    """Take the (N-dim) array and return a 16 bit version of it.
    """
    array_rescaled = np.zeros((array.shape), dtype=np.uint16)
    old_max, old_min = np.amax(array), np.amin(array)
    amplitude = (2**16 / (old_max - old_min))
    array_rescaled[:, :] = (array[:, :] - old_min) * amplitude
    return array_rescaled

def deconvolve_RL(stack, 
                  psf,
                  iterations,
                  tell_steps=False):
    """Simple Lucy-Richarson deconvolution.
    N.B. TO DO: add renormalization.
    N.B.2 It returns a real-valued result, not int.
    """
    o = np.copy(stack).astype(complex)
    # Richardson-Lucy in the for loop
    for k in range (iterations):
        step_0 = stack/(IFT3(FT3(o)*FT3(psf)))
        step_1 = IFT3(FT3(step_0)*np.conj(FT3(psf)))
        o *= step_1
        if(tell_steps): print(k)
    return np.real(o)

def gaussian_2D(dim, center=[0, 0], sigma=1):
    """ Just a 2D gaussian, cetered at origin.
        - dim = input extent 
        (assumed square, e.g. for a 1024x1024 image it should be simply 1024, 
         and the extent would go [-512;512])
        - sigma = stdev
    """
    x, y = np.meshgrid(np.arange(-dim/2, dim/2, 1), np.arange(-dim/2, dim/2, 1))
    top = (x - center[1])**2 + (y - center[0])**2 # row major convention
    return np.exp(-(top / (2 * sigma)**2))

def normalize_0_1(array):
    """ Normalize input (N-dim) array to 0-1 range.
    """
    maximum, minimum = np.amax(array), np.amin(array)
    normalized = np.zeros(array.shape)
    delta = maximum - minimum
    normalized = (array - minimum) / delta
    return normalized

def spatial_Xcorr_2D(f, g):
    """
    Cross-correlation between two 2D functions: (f**g).
    N.B. f can be considered as the moving input, g as the target.
    - inputs are padded to avoid artifacts (this makes it slower)
    - The output is normalized to [0,1)
    """
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
    spatial_cross = ft.ifftshift(ft.ifft2(ft.ifftshift(ONE) \
                  * np.conj(ft.ifftshift(TWO)))) \
                    [int(M/2) :int(M/2+M), int(N/2) : int(N/2+N)]
    #spatial_cross = normalize_0_1(spatial_cross)
    return np.real(spatial_cross)

def xcorr_peak_2D(cross):
    row_shift, col_shift = np.unravel_index(np.argmax(cross), cross.shape)
    return int(row_shift - cross.shape[0]/2), int(col_shift - cross.shape[1]/2)

def spatial_xcorr_3D(f, g):
    """
    Cross-correlation between two 2D functions: (f**g).
    N.B. f can be considered as the moving input, g as the target.
    - inputs are padded to avoid artifacts (this makes it slower)
    - The output is normalized to [0,1)
    """
    Z, M, N = f.shape[0], f.shape[1], f.shape[2]
    one, two = np.pad(np.copy(f),
                      ((int(Z/2), int(Z/2)),
                      (int(M/2), int(M/2)),
                       (int(N/2), int(N/2))),
                      mode = 'constant',
                      constant_values=(0, 0, 0)),\
               np.pad(np.copy(g),
                      ((int(z/2), int(z/2)),
                      (int(M/2), int(M/2)),
                      (int(N/2), int(N/2))),
                      mode = 'constant', 
                      constant_values=(0, 0, 0))                  
    ONE, TWO =   FT3(one), FT3(two)
    spatial_cross = ft.ifftshift(ft.ifftn(ft.ifftshift(ONE) \
                  * np.conj(ft.ifftshift(TWO)))) \
                    [int(Z/2) :int(Z/2+Z),
                    int(M/2) :int(M/2+M),
                    int(N/2) : int(N/2+N)]
    spatial_cross = normalize_0_1(spatial_cross)
    return np.real(spatial_cross)

def xcorr_peak_3D(cross):
    depth_shift, row_shift, col_shift = \
                                np.unravel_index(np.argmax(cross), cross.shape)
    return int(depth_shift - cross.shape[0]/2), \
           int(row_shift - cross.shape[0]/2), \
           int(col_shift - cross.shape[1]/2)

def remove_background(image, background_image):
    """ Subtracts the background mean. 
        - N.B. 16 bit input images
        - it takes care of negative values
    """
    background_average = np.mean(background_image)
    subtracted_image = np.zeros((image.shape), dtype=np.uint16)
    subtracted_image[:, :] = np.maximum(image[:, :] - background_average,\
                                        subtracted_image)
    return subtracted_image

def shift_image(image, shift):                 
    H, W = image.shape
    shifted = np.copy(image)
    if shift[0] >= 0: shifted[int(shift[0]):, :] = \
                                    image[:int(H - shift[0]), :] # shift up
    elif shift[0] < 0: shifted[:int(H + shift[0]), :] = \
                                     image[int(-shift[0]):, :] # shift down
    if shift[1] > 0:
        # shift right
        shifted[:, int(shift[1]):] = shifted[:, :int(W - shift[1])]
    elif shift[1] < 0:
        # shift right
        shifted[:, :int(W + shift[1])] = shifted[:, int(- shift[1]):]

    return shifted
    
def open_binary_volume_with_hotpixel_correction(name, 
                                                VOLUME_SLICES,
                                                IMAGES_DIMENSION,
                                                hotpixel_value=64000,
                                                format=np.uint16):
    """ It also performs hotpixel correction"""
    with open(name, 'rb') as file:
        raw_array = np.fromfile(file, dtype=format)
    raw_array[raw_array[:] > hotpixel_value] = 0
    volume_array =  np.reshape(raw_array, (VOLUME_SLICES,
                                           IMAGES_DIMENSION,
                                           IMAGES_DIMENSION))
    return volume_array

def files_names_list(total_volumes, seed_0='SPC00_TM', 
                                    seed_1='_ANG000_CM', 
                                    seed_2='_CHN00_PH0'):
    """ Basically used to list acquisition files so that I can parallelize.
    List paradigm: [entry (int), views_1 (array), views_2 (array)]
    """
    files_list = []
    j = 0
    for i in range(total_volumes):
        temp_list = [i]
        for k in range(0, 2):
            temp_list.append(seed_0 + f'{j:05}' + seed_1
                            + str(k) + seed_2 + ".stack")
        files_list.append(temp_list)
        j += 1
    return files_list

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

    print('\nI am working\n')
    gau1 = gaussian_2D(1024, sigma=20)
    # gau2 = gaussian_2D(1024, [-100, -100], 20) #row major convention

    # cross = spatial_xcorr_2D(gau2, gau1)
    # row, col = xcorr_peak_2D(cross)
    # print(row, col)
    # row = row * -1
    # col = col * -1
    # image = shift_image(gau2, (row, col))
    # plt.imshow(image)
    # plt.show() 