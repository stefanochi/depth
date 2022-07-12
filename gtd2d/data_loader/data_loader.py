import array
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_gt_depth(path, shape):
    """Load all the exr ground truth data with the timesteps
    f points to the folder containing depthmaps.txt, the file that contains the timesteps
    and the path to the correspoinding exr. It also contains the depthmaps folder containing the exrs"""
    gt_depths_f = np.genfromtxt(path + "depthmaps.txt", dtype="str")
    gt_depths = np.zeros((gt_depths_f.shape[0], shape[0], shape[1]))
    i = 0
    for t, d in gt_depths_f:
        gt_depths[i] = load_exr(path + d, shape)
        i += 1


    return gt_depths


def load_exr(f, shape):
    p = f
    # Open the input file
    file = OpenEXR.InputFile(p)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (Z) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("Z")]

    return np.array(Z).reshape(shape)
