from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import math_utils as matut
import raw_images_manipulation_utilities as rawman
import scipy as sp
from skimage.filters.rank import entropy
from skimage.morphology import disk


if __name__ == '__main__':

    # define the two fake images to blend
    # Assuming we have a cmaera with 1024x104 pixels
    # looking at a sample with 300nm pixel size,
    # using blue (488nm) light,
    # using an objective that has 1 NA, so that we can choose
    # to model the resolution with the Rayleigh formula:
    # D = 0.61 * lambda / NA
    resolution = .61 * .488
    M, N = 1024, 1024 # image dimensions
    image_1 = np.zeros((M, N))
    image_2 = np.zeros((M, N))
    image_true = np.zeros((M, N))
    sigma_1 = 8
    sigma_2 = 15
    margins = 100 # pixels to be excluded for point positioning
    number_of_points = 400
    points_x = np.random.randint(
        low= -M/2 + margins,
        high= M/2 - margins,
        size= number_of_points)
    points_y = np.random.randint(
        low= -N/2 + margins,
        high= N/2 - margins,
        size= number_of_points)
    for x, y in zip(points_x, points_y):
        sigma = np.random.choice([sigma_1, sigma_2])
        if sigma == sigma_1:
            sigma_b = sigma_2
        else:
            sigma_b = sigma_1
        image_1 += (10 / sigma) * matut.gaussian_2D(
           1024,
           [x, y],
           sigma)
        image_2 += (10 / sigma_b) * matut.gaussian_2D(
            1024, 
            [x, y],
            sigma_b)
        image_true += 100 * matut.gaussian_2D(1024, [x, y], 2)

    Gaus_1 = matut.gaussian_2D(M, [0,0], 2)
    Gaus_1 = matut.FT2(Gaus_1)

    # variables to explore:
    # - differences in the sigmas of the images
    # - differences in the number of points
    # - same workflow to compare using entropy of scipy

    diff = []
    diff_2 = []
    for i in range (120):
        Gaus_1 = matut.gaussian_2D(M, [0,0], i+1)
        Gaus_1 = matut.FT2(Gaus_1)

        w_1 = np.abs(image_1 - matut.IFT2((matut.FT2(image_1) * Gaus_1)))**2
        w_2 = np.abs(image_2 - matut.IFT2((matut.FT2(image_2) * Gaus_1)))**2
        diff.append(np.sum(np.abs(w_1/np.amax(w_1) - w_2/np.amax(w_2))))
        diff_2.append(np.sum(np.abs(w_1/np.amax(w_1+w_2) - w_2/np.amax(w_2+w_1))))
    
    plt.figure(1)
    plt.plot(diff, 'b.--', alpha = .5)
    plt.plot(diff_2, 'r.--')
    plt.figure(2)
    plt.imshow(image_1)

    # comparison between sigma=1, the best sigma (argmax) and '20', as Preibish
    
    position = np.argmax(diff_2[11:])+1
    print(np.argmax(diff_2[11:])+1)


    merged_1 = rawman.merge_two_images(image_1, image_2, sigma=1)
    merged_max = rawman.merge_two_images(image_1, image_2, sigma=position)
    merged_20 = rawman.merge_two_images(image_1, image_2, sigma=20)
    merged_original = rawman.merge_two_images(image_1, image_2, method='preib')

    plt.figure('sigma_1')
    plt.imshow(merged_1)
    plt.figure('sigma_max')
    plt.imshow(merged_max)
    plt.figure('sigma_20')
    plt.imshow(merged_20)
    plt.figure('original')
    plt.imshow(merged_original)
    plt.figure('diagonal comparison')
    # plt.plot(np.diag(merged_1), label='sigma = 1')
    plt.plot(np.diag(merged_max), label='sigma = argmax')
    plt.plot(np.diag(merged_20), label='sigma = 20')
    plt.plot(np.diag(merged_original), label='oritginal')
    plt.legend()
    plt.figure('truth')
    plt.imshow(image_true, cmap='gray')
    plt.show()
    print(np.amax(merged_original))
    print(np.amax(merged_max))
    print(np.std(merged_original))
    print(np.std(merged_max))