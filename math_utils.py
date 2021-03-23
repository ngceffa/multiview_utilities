import numpy as np 
from numba import jit
import matplotlib.pyplot as plt
import scipy as sp
from scipy.fftpack import fft, ifft, fft2, fftn, ifft2, ifftn, fftshift, ifftshift, fftfreq
import time

def IFT2(f):
    """ 2D Fourier Transform, with proper shift
    """
    return (fftshift(ifft2(ifftshift(f))))

def FT2(f):
    """ 2D Inverse Fourier Transform, with proper shift
    """
    return (fftshift(fft2(ifftshift(f))))

def FT3(f):
    """ 3D Fourier Transform, with proper shift
    """
    return fftshift(fftn(ifftshift(f)))

def IFT3(F):
    """ 3D Inverse Fourier Transform, with proper shift
    """
    return ifftshift(ifftn(fftshift(F)))

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
    # lucy-richardson in the for loop
    for k in range (iterations):
        step_0 = stack/(IFT3(FT3(o)*FT3(psf)))
        step_1 = IFT3(FT3(step_0)*np.conj(FT3(psf)))
        o *= step_1
        if(tell_steps):
            print(k)
    return np.real(o)

def gaussian_2D(dim, sigma):
    """ Just a 2D gaussian, cetered at origin.
        - dim = input extent 
        (assumed square, e.g. for a 1024x1024 image it should be simply 1024)
        - sigma = stdev
    """
    x, y = np.meshgrid(np.arange(-dim/2, dim/2, 1), np.arange(-dim/2, dim/2, 1))
    top = x**2+y**2
    return np.exp(-(top/(2 * sigma)**2))

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
    spatial_cross = ifftshift(ifft2(ifftshift(ONE) * np.conj(ifftshift(TWO))))\
                    [int(M/2) :int(M/2+M), int(N/2) : int(N/2+N)]
    spatial_cross = normalize_0_1(spatial_cross)
    return np.abs(spatial_cross)

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
    H, W = image.shape # Z = num. of images in stack; M, N = rows, cols;
    shifted = np.zeros((H, W),dtype = np.uint16)
    if shift[0] > 0: shifted[int(shift[0]):, :] = image[:int(H - shift[0]), :] # shift up
    elif shift[0] < 0: shifted[:int(H - shift[0]), :] = image[int(shift[0]):, :] # shift down
    if shift[1] < 0: shifted[:, :int(W - shift[1])] = shifted[:, int(shift[1]):] # shift left
    elif shift[1] > 0: shifted[:, int(shift[1]):] = shifted[:, :int(W - shift[1])] # shift right
    return shifted

if __name__ == '__main__':

    gau = gaussian_2D(1021, 10)*10
    start = time.time()
    gau_16 = normalize_0_1(gau)
    gau_s = shift_image(gau, [0, 50])
    print(time.time()-start)
    plt.imshow(gau_s)
    plt.show()