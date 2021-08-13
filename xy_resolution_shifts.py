import numpy as np
import matplotlib.pyplot as plt
import imageio as io
import math_utils as matut

# Given a binary image containing the target ROI,
# create multiple shifted copies, to test the resolution.

if __name__ == '__main__':

    PATH = '/home/ngc/Desktop/roitest' # where to find the image
    max_shift = 100
    shift_step = 10
    image_name = '/roi.tif'
    image = io.imread(PATH + image_name)

    shift_right = True
    shift_left = False
    shift_up = False
    shift_down = False

    for i in range(0, max_shift, shift_step):
        if(shift_up == True):
            up = matut.shift_image(image, [i + 1, 0])
            io.imwrite(PATH + '/up_' + str(i + 1) + '.png', up)
        if(shift_down == True):
            down = matut.shift_image(image, [-i - 1, 0])
            io.imwrite(PATH + '/down_' + str(i + 1) + '.png', down)
        if(shift_right == True):
            right = matut.shift_image(image, [0, i + 1])
            io.imwrite(PATH + '/right_' + str(i + 1) + '.png', right)
        if(shift_left == True):
            left = matut.shift_image(image, [0, -i - 1])
            io.imwrite(PATH + '/left_' + str(i + 1) + '.png', left)
            
