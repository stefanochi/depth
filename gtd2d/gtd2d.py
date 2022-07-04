import numpy as np

from .runner.python_runner import PythonRunner
from .data_loader import filter

def run():
    path = "C:/Users/schiavaz/lava-nc/depth/"
    dataset = "data/esim_data/curvy/"

    events = np.loadtxt(path + dataset + "events.txt")
    poses = np.loadtxt(path + dataset + "groundtruth.txt")
    poses[:, [1, 2, 3]] = poses[:, [2, 3, 1]]
    calib = np.loadtxt(path + dataset + "calib.txt")
    calib[0] = 200
    calib[1] = 200
    calib[2] = 120
    calib[3] = 90
    shape = (180, 240)
    chunk_size = 5000

    events = filter.filter_refract(events, 0.1)
    # events = filter.filter_time(events, 2.5, 2.7)

    runner = PythonRunner(events, poses, shape, chunk_size, calib)

    output = runner.run()
    return output

# run()