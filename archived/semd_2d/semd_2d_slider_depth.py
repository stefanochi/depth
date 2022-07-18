import sys

path = "C:/Users/schiavaz/lava-nc/depth/"
sys.path.append(path)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import cv2 as cv
from tqdm import tqdm

from lava.magma.core.run_configs import RunConfig

import events_utils.filter as flt
import events_utils.visualize as vis
import events_utils.time_difference as td
import events_utils.time_difference_2d as td2d
import events_utils.tum_vie as tum_dset

from scipy.spatial.transform import Rotation as R

import semd2d_utils

if __name__ == '__main__':
    events = np.loadtxt(path + "/data/slider-depth/events.txt")
    events = flt.filter_time(events, 1.5, 1.6)

    # use only the specified patch of input events
    patch_center = (94, 50)
    patch_size = 10
    events = semd2d_utils.filter_patch(events, patch_center, patch_size)

    data_steps = 20 # timesteps to divide the data into
    sim_steps = 20 # simulation steps
    shape = (patch_size, patch_size)
    t_start = events[0, 0]
    duration = events[-1, 0] - events[0, 0]

    events = flt.filter_refract(events, 0.1)

    args = {
        "shape":shape,
        "conv_shape": (2,2),
        "conv_stride": (2, 2),
        "thresh_conv": 0.5,
        "detector_du": 0.1
    }
    data_u, data_v, data_d = semd2d_utils.run_sim(args, events, data_steps, sim_steps)