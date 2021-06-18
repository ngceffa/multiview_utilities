import multiprocessing as mp
import os
import numpy as np
import lumped as lmp
import tifffile as tif
import matplotlib.pyplot as plt
import time
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift


def FT3(f):
    return (fftshift(fftn(ifftshift(f))))

def IFT3(f):
    return (fftshift(ifftn(ifftshift(f))))


def temporal_shift_exploration(file_path, static, moving):
	lmp.open_binary_stack(filepath + static, filep)
	return None

start = '/home/ngc/Data/16_06/fift_1_percent_agar_20210616_164944'
stk_start = '/SPC00_TM00000_ANG000_CM0_CHN00_PH0.stack'
end = '/home/ngc/Data/16_06/fift_1_percent_agar_25min_later_20210616_171307'
stk_end = '/SPC00_TM00000_ANG000_CM0_CHN00_PH0.stack'

background_0 = start + '/Background_0.tif'
background_1 = end + '/Background_0.tif'

now = time.time()
stack_0 = lmp.open_binary_stack(start+stk_start, background_0)
stack_1 = lmp.open_binary_stack(end+stk_end, background_1)
print(time.time() - now)

# explore_range = np.asarray((int(stack_0.shape[0] / 2) - 5,
# 			int(stack_0.shape[0] / 2) + 5))


F = FT3(stack_0[:, 800:1100, 1000:1300])
G = FT3(stack_1[:, 800:1100, 1000:1300])

corr = ifftshift(ifftn(ifftshift(F) * np.conj(ifftshift(G))))
corr = lmp.normalize_0_1(corr)

centre = (np.argmax(corr))
centre = np.unravel_index(centre, corr.shape)
print(centre)
