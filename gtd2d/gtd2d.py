import numpy as np

from runner.python_runner import PythonRunner
from runner.lava_runner import LavaRunner
from data_loader import filter
import pickle
import json
import sys
import time


def load_data(cfg):
    events = np.loadtxt(cfg["events_path"])
    poses = np.loadtxt(cfg["poses_path"])
    # TODO fix camera orientation in certain scenarios
    if cfg["shift_axes"]:
        poses[:, [1, 2, 3]] = poses[:, [2, 3, 1]]
    calib = np.loadtxt(cfg["calib_path"])
    shape = cfg["dvs_shape"]
    sub_factor = cfg["subsampling_factor"]
    refract_period = cfg["refractory_period"]
    time_range = cfg["time_range"]
    timesteps_second = cfg["timesteps_second"]
    # Perform the event data subsampling
    # Can also be done in lava entirely
    # TODO verify that the result is the same
    print("sub factor: {}".format(sub_factor))
    # DEBUG
    #events = events[events[:, 3] == 0]
    if sub_factor != 1:
        events, shape = filter.filter_conv(events, shape, factor=sub_factor)
        calib = calib / sub_factor

    events = filter.filter_refract(events, refract_period)
    events = filter.filter_time(events, time_range[0], time_range[1])

    return events, poses, calib, shape


def setup_lava(cfg):
    events, poses, calib, shape = load_data(cfg)
    timesteps_second = cfg["timesteps_second"]

    sequence_duration = events[-1, 0] - events[0, 0]
    print("sequqnece duration: ", sequence_duration)
    runner = LavaRunner(events, poses, shape, calib, timesteps_second * sequence_duration, cfg)
    print("Runner initialized: ")
    return runner


def setup_python(cfg):
    events, poses, calib, shape = load_data(cfg)

    chunk_size = cfg["chunk_size"]
    print(chunk_size)
    runner = PythonRunner(events, poses, shape, chunk_size, calib, cfg)

    return runner


def setup(cfg_file):
    # load the config file
    with open(cfg_file) as f:
        cfg = json.load(f)

    if cfg["use_lava"]:
        return setup_lava(cfg)
    else:
        return setup_python(cfg)


if __name__ == '__main__':
    assert len(sys.argv) > 1
    path = sys.argv[1]
    runner = setup(path)
    #try:
    start_time = time.time()
    out = runner.run()
    end_time = time.time()

    print("run time: {}".format(end_time - start_time))
    #except Exception as e:
     #   print(e)

    with open(runner.cfg["output_path"], 'wb') as f:
        pickle.dump(out, f)
