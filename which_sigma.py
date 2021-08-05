import numpy as np
import matplotlib.pyplot as plt
import math_utils as matut
import raw_images_manipulation_utilities as rim


if __name__ == '__main__':

    dimension = 1001

    x = np.arange(0, dimension, 1)
    y = np.arange(0, dimension, 1)
    x, y = np.meshgrid(x, y)

    spots_0 = np.zeros((1, dimension, dimension))
    spots_1 = np.zeros((1, dimension, dimension))

    # for i in range(300, 701, 100):
    #     for j in range(300, 701, 100):
    #         local = .1 * (i + j) * np.exp(
    #             -np.pi * (((x - i)**2 /(.05 * (i+j))**2) 
    #             + ((y - j)**2 / (.05 * (i+j))**2))
    #         )
    #         spots_0[0, :, :] += local

    # for i in range(700, 300, -100):
    #     for j in range(700, 300, -100):
    #         local = .1 * (i + j) * np.exp(
    #             -np.pi * (((x - i)**2 /(.05 * (i+j))**2) 
    #             + ((y - j)**2 / (.05 * (i+j))**2))
    #         )
    #         spots_1[0, :, :] += local
    
    spots_0 += 100 * np.exp(
                -np.pi * (((x - 500)**2 / 50**2) 
                + ((y - 250)**2 / 50**2))
            )
    spots_0 += 50 * np.exp(
                -np.pi * (((x - 500)**2 / 100**2) 
                + ((y - 400)**2 / 100**2))
            )
    spots_1 += 50 * np.exp(
                -np.pi * (((x - 500)**2 / 100**2) 
                + ((y - 250)**2 / 100**2))
            )
    spots_1 += 100 * np.exp(
                -np.pi * (((x - 500)**2 / 50**2) 
                + ((y - 400)**2 / 50**2))
            )

    merged = []
    # plt.figure('plots')
    # for i in range(10):
    #     merged.append(rim.merge_views(
    #     spots_0,
    #     spots_1,
    #     method='without_sigma_2',
    #     sigma_1= 2 * i + 10
    #     ))
    #     plt.plot(merged[i][0, :, 500], label = str(i))
    # plt.legend()
    # plt.show()
    merged.append(rim.merge_views(
        spots_0,
        spots_1,
        method='without_sigma_2',
        sigma_1=45
        ))
    merged.append(rim.merge_views(
        spots_0,
        spots_1,
        method='more_deriv',
        sigma_1=45
        ))
    plt.plot(merged[0][0, :, 500], label='sigma')
    plt.plot(merged[1][0, :, 500], label='deriv')
    plt.legend()
    plt.show()
    # figure = plt.figure('starting images')
    # figure.add_subplot(231)
    # plt.imshow(spots_0[0, :, :])
    # figure.add_subplot(232)
    # plt.imshow(spots_1[0, :, :])
    # figure.add_subplot(233)
    # plt.imshow(merged[0, :, :])
    # figure.add_subplot(235)
    # plt.plot(merged[0, :, 500], label = 'final')
    # plt.plot(spots_0[0, :, 500], label = '0')
    # plt.plot(spots_1[0, :, 500], label = '1')
    # plt.legend()
    # plt.show()
