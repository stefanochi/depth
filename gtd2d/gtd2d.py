import numpy as np

from runner.python_runner import PythonRunner
from runner.lava_runner import LavaRunner
from data_loader import filter
import pickle
import json
import sys


def setup_lava(cfg):
    events = np.loadtxt(cfg["events_path"])
    poses = np.loadtxt(cfg["poses_path"])
    # TODO fix camera orientation in certain scenarios
    # poses[:, [1, 2, 3]] = poses[:, [2, 3, 1]]
    calib = np.loadtxt(cfg["calib_path"])
    shape = cfg["dvs_shape"]
    sub_factor = cfg["subsampling_factor"]
    refract_period = cfg["refractory_period"]
    time_range = cfg["time_range"]
    timesteps_second = cfg["timesteps_second"]
    # Perform the event data subsampling
    # Can also be done in lava entirely
    # TODO verify that the result is the same
    if sub_factor != 1:
        events, shape = filter.filter_conv(events, shape, factor=sub_factor)
        calib = calib / sub_factor

    events = filter.filter_refract(events, refract_period)
    events = filter.filter_time(events, time_range[0], time_range[1])

    sequence_duration = time_range[1] - time_range[0]
    runner = LavaRunner(events, poses, shape, calib, timesteps_second * sequence_duration, cfg)
    print("Runner initialized: ")
    temp = vars(runner)
    for item in temp:
        print(item, ':', temp[item])

    return runner


def setup_python(p, run_lava=False):
    # print(events.shape)
    # if run_lava:
    #     runner = LavaRunner(events, poses, shape, calib, 200, cfg)
    # else:
    #     runner = PythonRunner(events, poses, shape, chunk_size, calib)
    #
    # return runner
    return


def setup(cfg_file, run_lava=False):
    # load the config file
    with open(cfg_file) as f:
        cfg = json.load(f)

    if run_lava:
        return setup_lava(cfg)
    else:
        return setup_python(cfg)


if __name__ == '__main__':
    assert len(sys.argv) > 1
    path = sys.argv[1]
    runner = setup(path, run_lava=True)
    out = runner.run()

    with open(runner.cfg["output_path"], 'wb') as f:
        pickle.dump(out, f)
