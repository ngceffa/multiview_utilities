import numpy as np
import matplotlib.pyplot as plt
import math_utils as matut
plt.style.use('seaborn-dark')

if __name__ == '__main__':

    x = np.arange(0, 100, .05)
    signal_1 = np.zeros((len(x)))
    signal_2 = np.zeros((len(x)))

    peaks_1 = np.random.uniform(20, 80, 10)
    peaks_2 = np.random.uniform(20, 80, 10)
    widths_1 = np.random.uniform(1, 6, 10)
    widths_2 = 7 - widths_1

    for i in range(len(peaks_1)):
        signal_1 += matut.gaus(x, peaks_1[i], widths_1[i]) * (7 - widths_1[i])
        signal_2 += matut.gaus(x, peaks_1[i], widths_2[i]) * (7 - widths_2[i])

    signal_1 += np.random.rand(signal_1.shape[0]) * .0
    signal_2 += np.random.rand(signal_2.shape[0]) * .0


    # center_0 = 46
    # center_1 = 50
    # signal_1 += matut.gaus(x, center_0, 4) * .4
    # signal_1 += matut.gaus(x, center_1, 2)
    # signal_2 += matut.gaus(x, center_0, 2)
    # signal_2 += matut.gaus(x, center_1, 4) * .4
# ----------------------------------------------------------------------
# average merge
    average = (signal_1 + signal_2) / 2
    # result_average = plt.figure('average')
    # result_average.add_subplot(211)
    # plt.plot(x, signal_1, 'k-')
    # plt.plot(x, signal_2, 'r-')
    # result_average.add_subplot(212)
    # plt.plot(average)

# ----------------------------------------------------------------------
# preibish original
    gaus_1 = matut.gaus(x, 0, 20)
    gaus_2 = matut.gaus(x, 0, 40)
    Gaus_1 = matut.FT(gaus_1)
    Gaus_2 = matut.FT(gaus_2)
    Signal_1 = matut.FT(signal_1)
    Signal_2 = matut.FT(signal_2)
    top_1 = np.mean(signal_1)
    top_2 = np.mean(signal_2)

    weight_1 = (signal_1 - matut.IFT((Signal_1 * Gaus_1)))**2
    Weight_1 = matut.FT(weight_1)
    weight_1 = matut.IFT(Weight_1 * Gaus_2)

    weight_2 = (signal_2 - matut.IFT((Signal_2 * Gaus_1)))**2
    Weight_2 = matut.FT(weight_2)
    weight_2 = matut.IFT(Weight_2 * Gaus_2)

    tot =  weight_1 + weight_2
    preib_merged = np.real((weight_1 * signal_1 
                                + weight_2 * signal_2)\
                                / tot)
    
    gaus_1 = matut.gaus(x, 0, .5)
    gaus_2 = matut.gaus(x, 0, 1)
    Gaus_1 = matut.FT(gaus_1)
    Gaus_2 = matut.FT(gaus_2)
    Signal_1 = matut.FT(signal_1)
    Signal_2 = matut.FT(signal_2)
    top_1 = np.mean(signal_1)
    top_2 = np.mean(signal_2)

    weight_1 = (signal_1 - matut.IFT((Signal_1 * Gaus_1)))**2
    Weight_1 = matut.FT(weight_1)
    # weight_1 = matut.IFT(Weight_1 * Gaus_2)

    weight_2 = (signal_2 - matut.IFT((Signal_2 * Gaus_1)))**2
    Weight_2 = matut.FT(weight_2)
    # weight_2 = matut.IFT(Weight_2 * Gaus_2)

    extract_1 = weight_1 > weight_2

    tot =  weight_1 + weight_2
    preib_merged_10 = np.real((extract_1 * signal_1 
                                + (1 - extract_1) * signal_2)\
                                )
    
    # result_preib = plt.figure('preib')
    # result_preib.add_subplot(211)
    # plt.plot(x, signal_1, 'k-')
    # plt.plot(x, signal_2, 'r-')
    # result_preib.add_subplot(212)
    # plt.plot(preib_merged)

# ----------------------------------------------------------------------
#   no_sigma_2

    # gaus_1 = matut.gaus(x, 0, 5)
    # Gaus_1 = matut.FT(gaus_1)

    # weight_1 = np.abs(signal_1 - matut.IFT((Signal_1 * Gaus_1)))

    # weight_2 = np.abs(signal_2 - matut.IFT((Signal_2 * Gaus_1)))

    # tot = weight_1 + weight_2
    # only_sigma = np.real((weight_1 * signal_1 
    #                             + weight_2 * signal_2)\
    #                             / tot)
    # gaus_1 = matut.gaus(x, 0, 100)
    # Gaus_1 = matut.FT(gaus_1)

    # weight_1 = np.abs(signal_1 - matut.IFT((Signal_1 * Gaus_1)))

    # weight_2 = np.abs(signal_2 - matut.IFT((Signal_2 * Gaus_1)))

    # tot = weight_1 + weight_2
    # only_sigma_5 = np.real((weight_1 * signal_1 
    #                             + weight_2 * signal_2)\
    #                             / tot)
    # result_no_sigma = plt.figure('no_sigma')
    # result_no_sigma.add_subplot(211)
    # plt.plot(x, signal_1, 'k-')
    # plt.plot(x, signal_2, 'r-')
    # result_no_sigma.add_subplot(212)
    # plt.plot(only_sigma)
    # plt.show()
# ----------------------------------------------------------------------
#   derivative

    gaus_1 = matut.gaus(x, 0, 20)
    gaus_2 = matut.gaus(x, 0, 40)
    Gaus_1 = matut.FT(gaus_1)
    Gaus_2 = matut.FT(gaus_2)
    Signal_1 = matut.FT(signal_1)
    Signal_2 = matut.FT(signal_2)
    top_1 = np.mean(signal_1)
    top_2 = np.mean(signal_2)

    weight_1 = (signal_1 - matut.IFT((Signal_1 * Gaus_1)))**2
    Weight_1 = matut.FT(weight_1)
    # weight_1 = np.real(matut.IFT(Weight_1 * Gaus_2))

    weight_2 = (signal_2 - matut.IFT((Signal_2 * Gaus_1)))**2
    Weight_2 = matut.FT(weight_2)
    # weight_2 = np.real(matut.IFT(Weight_2 * Gaus_2))

    tot =  weight_1 + weight_2

    # weight_1 += np.abs(np.gradient(signal_1))/ np.abs(np.gradient(signal_1))
    # weight_2 += np.abs(np.gradient(signal_2)) / np.abs(np.gradient(signal_2))
    winner = weight_1 > weight_2
    plt.plot(weight_1 > weight_2)
    gaus_1 = matut.gaus(x, 0, 1)
    Gaus_1 = matut.FT(gaus_1)
    winner = np.abs(matut.IFT(matut.FT((weight_1 > weight_2)) * Gaus_1))
    plt.plot(winner, 'r')
    plt.show()
    print(winner)
    tot = weight_1 + weight_2
    deriv = np.real((winner * signal_1 
                                + ((np.amax(winner) - winner) * signal_2)))
    # result_no_sigma = plt.figure('no_sigma')
    # result_no_sigma.add_subplot(211)
    # plt.plot(x, signal_1, 'k-')
    # plt.plot(x, signal_2, 'r-')
    # result_no_sigma.add_subplot(212)
    # plt.plot(only_sigma)
    # plt.show()
        
# ----------------------------------------------------------------------
# preibish + derivative
    # gaus_1 = matut.gaus(x, 0, 10)
    # gaus_2 = matut.gaus(x, 0, 1)
    # Gaus_1 = matut.FT(gaus_1)
    # Gaus_2 = matut.FT(gaus_2)
    # Signal_1 = matut.FT(signal_1)
    # Signal_2 = matut.FT(signal_2)

    # # DO THE RESCALING CORRECTLY! THIS REDUCES DIFFERENCES!

    # weight_1 = (signal_1 - matut.IFT((Signal_1 * Gaus_1)))**2
    # Weight_1 = matut.FT(weight_1)
    # weight_1 = np.real(matut.IFT(Weight_1 * Gaus_2))

    # weight_2 = (signal_2 - matut.IFT((Signal_2 * Gaus_1)))**2
    # Weight_2 = matut.FT(weight_2)
    # weight_2 = np.real(matut.IFT(Weight_2 * Gaus_2))

    # w_1_max = np.amax(weight_1)
    # w_2_max = np.amax(weight_2)
    # w_1_min = np.amin(weight_1)
    # w_2_min = np.amin(weight_2)
    # # weight_1 = weight_1 / np.amax(weight_1)
    # # weight_2 = weight_2 / np.amax(weight_2)
    
    # grad_1 = np.abs(np.gradient(signal_1))
    # grad_2 = np.abs(np.gradient(signal_2))
    # g_1_max = np.amax(grad_1)
    # g_2_max = np.amax(grad_2)
    # g_1_min = np.amin(grad_1)
    # g_2_min = np.amin(grad_2)

    # grad_weight_1 = (w_1_max - w_1_min) \
    #                 * (grad_1 - g_1_min) \
    #                 / (g_1_max - g_1_min) + w_1_min
    # grad_weight_1[grad_weight_1==0] += w_1_max
    # grad_weight_2 = (w_2_max - w_2_min) \
    #                 * (grad_2 - g_2_min) \
    #                 / (g_2_max - g_2_min) + w_2_min

    # grad_weight_2[grad_weight_2==0] += w_2_max

    # weight_1 +=  .5 * grad_weight_1 
    # weight_2 +=  .5 * grad_weight_2

    # tot = weight_1 + weight_2
    # preib_deriv = np.real((weight_1 * signal_1 
    #                             + weight_2 * signal_2)\
    #                             / tot)
    
    final_comparinson = plt.figure('final')
    final_comparinson.add_subplot(211)
    plt.plot(x, signal_1, 'k-')
    plt.plot(x, signal_2, 'r-')
    final_comparinson.add_subplot(212)
    # plt.plot(x, average, 'y', label='average', alpha=.9, lw=6)
    plt.plot(x, preib_merged / np.amax(preib_merged), 'g', label='preib', alpha=.8)
    plt.plot(x, deriv / np.amax(deriv), 'b', label='deriv', alpha=.8)
    # plt.plot(x, preib_merged, 'g', label='original', alpha=.4)
    # plt.plot(x, preib_merged_10, 'b', label='10', alpha=.4)
    # plt.plot(x, preib_merged_5, 'm', label='5', alpha=.4)
    # plt.plot(x, preib_deriv, 'k', label='preib_deriv', alpha=.8)
    # plt.plot(x, only_sigma_5, 'r', label='5', alpha=.7)
    plt.legend()
    plt.show()


    #grad_front = (d_front[0]**2 + d_front[1]**2)
