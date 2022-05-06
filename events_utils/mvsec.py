import numpy as np
import hdf5plugin
import h5py

from scipy.spatial.transform import Rotation as R


def load_events(f, row_start=0, row_end=-1):
    data = h5py.File(f)
    events_h5 = data['davis']['left']['events']

    t = np.array(events_h5[row_start:row_end, 2])
    x = np.array(events_h5[row_start:row_end, 0])
    y = np.array(events_h5[row_start:row_end, 1])
    p = np.array(events_h5[row_start:row_end, 3])


    events = np.zeros((t.size, 4))
    events[:, 0] = t
    events[:, 1] = x
    events[:, 2] = y
    events[:, 3] = p

    return events

def _searchsorted(v, l):
    id_start = 0
    id_end = l.size -1
    print("val: {}".format(v))

    while True:
        # print("start id:{} val:{}".format(id_start, l[id_start]))
        # print("end id:{} val:{}".format(id_end, l[id_end]))
        # print("------")
        curr_id = int((id_end - id_start) / 2) + id_start
        if curr_id == id_end or curr_id == id_start:
            return curr_id
        if l[curr_id] == v:
            return curr_id
        if l[curr_id] < v:
            id_start = curr_id
        if l[curr_id] > v:
            id_end = curr_id

def load_events_time(f, t_start, t_end):
    # load the events in a specific time range. The times are in seconds
    data = h5py.File(f)
    events_h5 = data['davis']['left']['events']

    # print(events_h5[:,2].max() - events_h5[:, 2].min())
    # print(events_h5[:, 2].min())
    # print(events_h5[:, 2].size)

    id_start = _searchsorted(events_h5[0,2] + t_start, events_h5[:, 2])
    id_end = _searchsorted(events_h5[0,2] + t_end, events_h5[:, 2])


    print(id_start)
    print(id_end)
    print("{} events".format(id_end - id_start))

    return load_events(f, id_start, id_end)

def load_poses(f):
    data_gt = h5py.File(f)

    pose_data = np.array(data_gt['davis']['left']['pose'])
    pose_t = np.array(data_gt['davis']['left']['pose_ts'])

    gt_v = np.zeros((pose_data.shape[0], 8))

    gt_v[:, 0] = pose_t
    gt_v[:, 1:4] = pose_data[:, :3, 3]
    rs = R.from_matrix(pose_data[:, :3, :3])
    gt_v[:, 4:] = rs.as_quat()

    return gt_v

def load_odom(f):
    data = np.load(f)

    odom = np.zeros((data['lin_vel'].shape[0], 7))

    odom[:, 0] = data['timestamps']
    odom[:, 1:4] = data['lin_vel']
    odom[:, 4:] = data['ang_vel']

    odom[:, 1] = np.convolve(odom[:, 1], np.ones(10), mode="same")
    odom[:, 2] = np.convolve(odom[:, 2], np.ones(10), mode="same")
    odom[:, 3] = np.convolve(odom[:, 3], np.ones(10), mode="same")
    odom[:, 4] = np.convolve(odom[:, 4], np.ones(10), mode="same")
    odom[:, 5] = np.convolve(odom[:, 5], np.ones(10), mode="same")
    odom[:, 6] = np.convolve(odom[:, 6], np.ones(10), mode="same")

    return odom