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


if __name__ == '__main__':
    x = np.arange(1, 10, 1)