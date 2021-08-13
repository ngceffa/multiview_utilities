import numpy as np
import matplotlib.pyplot as plt
import math_utils as matut
import importlib
importlib.reload(matut);

if __name__ == '__main__':

    gaus_center = matut.gaussian_2D(512, sigma=5)
    gaus_moved = matut.gaussian_2D(512, [-80, 80], sigma=5)

    cross = matut.spatial_Xcorr_2D(gaus_moved, gaus_center)

    plt.figure('first image')
    plt.imshow(cross)

    shifted = matut.shift_image(gaus_center, [-80, 80])

    plt.figure('shifted')
    plt.imshow(shifted)
    plt.show()