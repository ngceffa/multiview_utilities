import math_utils as utils
import numpy as np


def cameras_shift_registration(views, shift):
    """ Cure the intrinsic shift between the two cameras by moving one set of 
        data by the shift found with a 2D cross-correlation.
    """
    shifted = np.zeros((views.shape), dtype=np.uint16)
    for i in range(views.shape[0]):
        shifted[i, :, :] = utils.shift_image(views[i, : :], shift)
    return shifted


if __name__ == '__main__':
    x = np.arange(1, 10, 1)