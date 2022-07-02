from os import listdir
from os.path import isfile, join
import numpy as np


def load_scn(folder_path, shape=(240, 320)):
    """
    Load the data of a sequence from: http://ebvds.neurocomputing.systems/EBSLAM3D/index.html
    :param folder_path: the path the folder containing the data
    :param shape: the resolution of the rgb camera
    :return: the events from the edvs, the depth gt for each event(different resolution?), the rgb images
    """
    # Provided data is: PrimeSense image coordinates (x,y), PrimeSense depth measurement,
    # EDVS image coordinates (x,y), Timestamp (in microsenconds), EDVS parity flag
    events_depths = np.loadtxt(folder_path + "events.tsv", delimiter="\t")
    events_depths[:, 5] *= 1e-6
    # events only
    events = np.copy(events_depths[:, 3:])
    events[:, [0, 1, 2]] = events[:, [2, 0, 1]]
    # depths only
    depths = np.copy(events_depths[:, [5, 0, 1, 2]])
    #depths[:, [0, 3]] = depths[:, [3, 0]]

    rgb_files = [join(folder_path + "/rgb/", f) for f in listdir(folder_path + "/rgb/")]

    rgb_imgs = np.zeros((len(rgb_files), shape[0], shape[1]), dtype="int8")

    for i, f in enumerate(rgb_files):
        img = np.fromfile(f, dtype="int8")
        img = img.reshape(shape[0], shape[1], 3)
        rgb_imgs[i] = rgb2gray(img)

    return events, depths, rgb_imgs


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
