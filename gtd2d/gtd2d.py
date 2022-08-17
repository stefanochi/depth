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
    groundtruth_available = True
    try:
        poses = np.loadtxt(cfg["poses_path"])
    except Exception as e:
        print("no groundtruth poses")
        groundtruth_available = False
        poses = [[0, 0, 0, 0, 0, 0, 0, 0]]

    if cfg["use_imu"]:
        no_imu = False
        try:
            imu = np.loadtxt(cfg["imu_path"])
        except Exception as e:
            print("no IMU file found. using the ground truth instead")
            cfg["use_imu"] = False
            no_imu = True
            if no_imu and not groundtruth_available:
                print("no position data available: exiting")
                sys.exit(1)

        if not no_imu:
            dt = (imu[1:, 0] - imu[:-1, 0]).mean()

            a = 1
            if groundtruth_available:
                gt_txyz = poses[:, :4]
                vel_xyz_gt = (gt_txyz[a:, 1:] - gt_txyz[:-a, 1:]) / (gt_txyz[a:, 0] - gt_txyz[:-a, 0])[:, None]
            else:
                vel_xyz_gt = np.array([[0, 0, 0]])

            imu_txyz = imu[:, :4]
            imu_txyz[:,3] -= 9.81
            initial_vel = vel_xyz_gt[0]
            imu_xyz_init = np.vstack([initial_vel, imu_txyz[:, 1:] * dt])
            vel_xyz_imu = np.cumsum(imu_xyz_init, axis=0)

            imu_vel = np.c_[imu[:, 0], vel_xyz_imu[1:]]
        else:
            imu_vel = None
    else:
        imu_vel = None

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
    print("sub factor: {}".format(sub_factor))
    # DEBUG
    #events = events[events[:, 3] == 0]
    if sub_factor != 1:
        events, shape = filter.filter_conv(events, shape, factor=sub_factor)
        calib = calib / sub_factor

    events = filter.filter_refract(events, refract_period)
    events = filter.filter_time(events, time_range[0], time_range[1])

    return events, poses, calib, shape, imu_vel


def setup_lava(cfg):
    events, poses, calib, shape, imu_vel = load_data(cfg)
    timesteps_second = cfg["timesteps_second"]

    sequence_duration = events[-1, 0] - events[0, 0]
    print("sequqnece duration: ", sequence_duration)
    runner = LavaRunner(events, poses, shape, calib, timesteps_second * sequence_duration, imu_vel, cfg)
    print("Runner initialized: ")
    return runner


def setup_python(cfg):
    events, poses, calib, shape, imu_vel = load_data(cfg)

    chunk_size = cfg["chunk_size"]
    timesteps_second = cfg["timesteps_second"]
    sequence_duration = events[-1, 0] - events[0, 0]
    chunk_size = events.shape[0] / (timesteps_second * sequence_duration)
    print("chunk size: ", chunk_size)
    runner = PythonRunner(events, poses, shape, chunk_size, calib, imu_vel, cfg)

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

    start_time = time.time()
    out = runner.run()
    end_time = time.time()

    print("run time: {}".format(end_time - start_time))

    with open(runner.cfg["output_path"], 'wb') as f:
        pickle.dump(out, f)
