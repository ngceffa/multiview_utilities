import numpy as np
import matplotlib.pyplot as plt

class Mymage(object):
    def __init__(
        self,
        filepath,
        num_stacks,
        images_per_stack,
        z_dim,
        x_dim,
        y_dim,
        dtype
        ):
        print('Creating files list')