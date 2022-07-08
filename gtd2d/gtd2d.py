import numpy as np

from runner.python_runner import PythonRunner
from runner.lava_runner import LavaRunner
from data_loader import filter
import pickle


def setup(p, run_lava=False):
    events = np.loadtxt(p + "events.txt")
    poses = np.loadtxt(p + "groundtruth.txt", max_rows=100000)
    # poses[:, [1, 2, 3]] = poses[:, [2, 3, 1]]
    calib = np.loadtxt(p + "calib.txt")
    calib[0] = 200
    calib[1] = 200
    calib[2] = 120
    calib[3] = 90
    shape = (180, 240)
    chunk_size = 1000
    sub_factor = 2

    events, shape = filter.filter_conv(events, shape, factor=sub_factor)
    calib = calib / sub_factor
    events = filter.filter_refract(events, 0.1)
    events = filter.filter_time(events, 0.1, 0.2)
    print(events.shape)
    if run_lava:
        runner = LavaRunner(events, poses, shape, calib, 20)
    else:
        runner = PythonRunner(events, poses, shape, chunk_size, calib)

    return runner


if __name__ == '__main__':
    path = "C:/Users/schiavaz/lava-nc/depth/"
    dataset = "data/esim_data/line_top/"
    runner = setup(path + dataset, run_lava=True)
    out = runner.run()

    with open(path + dataset + 'result.pkl', 'wb') as f:
        pickle.dump(out, f)
