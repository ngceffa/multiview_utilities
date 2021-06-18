import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift, ifftshift
import tifffile as tif

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

def deconvolve(image, psf, iterations=5):
    # object refers to the estimated "true" sample
    object = np.copy(image).astype(complex)
    # lucy-richardson in the for loop
    for k in range (iterations):
        step_0 = image / (IFT2(FT2(object) * FT2(psf)))
        step_1 = IFT2(FT2(step_0) * np.conj(FT2(psf)))
        object *= (step_1)**2
    return np.real(object)

def psf_gauss_2D(dimension, sigmas):
    x, y, = np.meshgrid(np.arange(0, dimension, 1), 
                        np.arange(0, dimension, 1))
    return np.exp(-( ((x - dimension / 2.)**2 / (2 * sigmas[0]**2)) 
                    + ((y - dimension / 2.)**2 / (2 * sigmas[1]**2))))

# yes
def FT2(f):
    return (fftshift(fft2(ifftshift(f))))

#yes
def IFT2(f):
    return (fftshift(ifft2(ifftshift(f))))

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    filepath = '/home/ngc/Data/20210614/20X_488_20210614_151940'
    stack = '/SPC00_TM00000_ANG000_CM0_CHN00_PH0.stack'
    background_name = '/Background_1.tif'
    slice = 3
    sigmas = [1.5, 1.5]
    psf = psf_gauss_2D(2304, sigmas)

    stack = open_binary_stack(filepath + stack, filepath+  background_name)

    deconv = deconvolve(stack[slice, :, :], psf, 4)

    # renormalize to uint16?
    deconv = deconv.astype(np.uint16)

    print(np.amax(deconv))
    print(np.amin(deconv))

    tif.imwrite('/home/ngc/Data/exp_2_iter_4_img_3.tif', deconv)
