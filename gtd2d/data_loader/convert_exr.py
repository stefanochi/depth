import numpy as np
import sys

import data_loader

"""Script to convert the exr files to a numpy array
There are some problems reading the exrs from windows, run on linux"""

path = sys.argv[1]

depths_gt = data_loader.load_gt_depth(path, (180, 240))

np.save(path + "/gt_depths.npy", depths_gt)
