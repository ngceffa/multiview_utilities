from re import I
import sys
from threading import stack_size
import tifffile
sys.path.append('/home/ngc/Documents/GitHub/MicrobeadsToPSF')
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import lumped as utils
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import time
#import importlib
#importlib.reload(psf)

def psf_gauss_2D(dimension, sigmas):
    x, y, = np.meshgrid(np.arange(0, dimension, 1), 
                        np.arange(0, dimension, 1))
    return np.exp(-( ((x - dimension / 2.)**2 / (2 * sigmas[0]**2)) 
                    + ((y - dimension / 2.)**2 / (2 * sigmas[1]**2))))

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

def gauss_3d(coords, bg, A_coeff, x_0, y_0, z_0, x_sig, y_sig, z_sig):
    # Expose X, Y, and Z coordinates
    z_val, x_val, y_val = coords

    # General Gaussian function 3D
    return bg + A_coeff * np.exp(
        -(
            (x_val - x_0) ** 2 / (2 * x_sig ** 2)
            + (y_val - y_0) ** 2 / (2 * y_sig ** 2)
            + (z_val - z_0) ** 2 / (2 * z_sig ** 2)
        )
    )

def fit_PSFs(
            img_path,
            background_path,
            x_box=4,
            y_box=4,
            z_box=8,
            bg=1e2,
            A_coeff=1e4,
            x_sig=2,
            z_sig=2,
            threshold=100,
            show=False
            ):

    beads = open_binary_stack(
        img_path,
        background_path
        )
    size_z, size_x, size_y = beads.shape

    x_val = np.arange(0, 2 * x_box + 1, 1)
    y_val = np.arange(0, 2 * y_box + 1, 1)
    z_val = np.arange(0, 2 * z_box + 1, 1)
    z_grid, x_grid, y_grid = np.meshgrid(z_val, x_val, y_val, indexing="ij")

    xdata = np.vstack((z_grid.ravel(), x_grid.ravel(), y_grid.ravel()))

    # Initial guesses
    y_sig = x_sig
    p0 = bg, A_coeff, x_box, y_box, z_box, x_sig, y_sig, z_sig

    # Initializations
    x_coord_abs = np.zeros(threshold, dtype=np.uint8)
    y_coord_abs = np.zeros(threshold, dtype=np.uint8)
    z_coord_abs = np.zeros(threshold, dtype=np.uint8)

    x_sigma = np.zeros(threshold)
    y_sigma = np.zeros(threshold)
    z_sigma = np.zeros(threshold)

    RMSE = np.zeros(threshold)

    x_sigma_values = []
    y_sigma_values = []
    z_sigma_values = []

    beads_target = np.copy(beads)

    beads_target[: z_box+1, :, :] = 0
    beads_target[beads.shape[0] - (z_box + 1), :, :] = 0
    beads_target[:, :x_box+1, :] = 0
    beads_target[:, beads.shape[0] - (x_box + 1), :] = 0
    beads_target[:, :, :y_box+1] = 0
    beads_target[:, :, beads.shape[0] - (y_box + 1)] = 0

    target = beads_target[25, :, :]

    for count in range(threshold):
        target = beads_target[25, :, :]
        z_coord = 25
        # Find first maximum in stack
        x_coord, y_coord = np.unravel_index(
            np.argmax(target), target.shape)
        z_coord_abs[count] = 2
        x_coord_abs[count] = int(x_coord)
        y_coord_abs[count] = int(y_coord)
        print(z_coord_abs[count], x_coord_abs[count],y_coord_abs[count])


        # Index in range? Otherwise put to type_min (edge cases)
        x_limits = x_coord_abs[count] > x_box and x_coord_abs[count] < size_x - 1 - x_box
        y_limits = y_coord_abs[count] > y_box and y_coord_abs[count] < size_y - 1 - y_box
        z_limits = z_coord_abs[count] > z_box and z_coord_abs[count] < size_z - 1 - z_box

        if x_limits and y_limits and z_limits:
            print('y')
            # Retrieve bead bounding box (the box doesn't have ownership in
            # Python, if beads is modified, box will be modified as well)
            box = beads[
                z_coord - z_box : z_coord + z_box + 1,
                x_coord - x_box : x_coord + x_box + 1,
                y_coord - y_box : y_coord + y_box + 1,
            ]

            # 3D Gaussian fit
            popt, pcov = curve_fit(gauss_3d, xdata, box.ravel(), p0)

            x_coord_abs[count] = int(popt[2] + x_coord - x_box)
            y_coord_abs[count] = int(popt[3] + y_coord - y_box)
            z_coord_abs[count] = int(popt[4] + z_coord - z_box)

            x_sigma[count] = popt[5]
            y_sigma[count] = popt[6]
            z_sigma[count] = popt[7]

            if popt[5] > 0 and popt[6] > 0 and popt[7] > 0:
                x_sigma_values.append(round(popt[6], 3))
                y_sigma_values.append(round(popt[7], 3))
                z_sigma_values.append(round(popt[5], 3))

            # Clear box
            beads_target[
                z_coord - z_box : z_coord + z_box,
                x_coord - x_box : x_coord + x_box,
                y_coord - y_box : y_coord + y_box
                ] = 0
        else:
            beads_target[z_coord - z_box : z_coord+ z_box + 1,
                x_coord - x_box : x_coord + x_box + 1,
                y_coord - y_box : y_coord + y_box + 1] = 0

    # Average values to obtain mean PSF shape.
    # (deconv. will use this values to create 2D or 3D gaussian PSF)
    average_x, average_y, average_z = (
                                       np.mean((x_sigma_values)),
                                       np.mean(y_sigma_values),
                                       np.mean(z_sigma_values),
                                      )
    if show == True:
        print(len(x_sigma_values))
        print(f'\naverage_x: {average_x}')
        print(f'\naverage_y: {average_y}')
        print(f'\naverage_z: {average_z}\n')
        plt.figure('recap')
        a = np.arange(0, len(x_sigma_values), 1)
        plt.scatter(a, x_sigma_values, label='x')
        plt.plot(a, x_sigma_values, label='x')
        plt.scatter(a, y_sigma_values, label='y')
        plt.plot(a, y_sigma_values, label='y')
        plt.scatter(a, z_sigma_values, label='z')
        plt.plot(a, z_sigma_values, label='z')
        plt.legend()
        plt.show()
    return average_x, average_y, average_z

# ------------------------------------------------------------------------------

filepath = '/home/ngc/Data/Beads20X_LowerC/CloserToDet2_20210607_165040'
background_path = filepath
image_name = '/SPC00_TM00000_ANG000_CM0_CHN00_PH0.stack'
background_name = '/Background_1.tif'
start = time.time()
x, y, z = fit_PSFs(
            filepath+image_name,
            filepath+background_name,
            threshold=20,
            show=True)

# the sigmas are saved in pixel units
sigmas = np.asarray([x, y, z])
np.save('sigmas', sigmas)
sigmas = np.load('sigmas.npy')

psf = psf_gauss_2D(2304, sigmas)
print(psf.shape)
plt.figure()
plt.imshow(psf)
plt.show()
