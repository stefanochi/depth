import numpy as np
import hdf5plugin
import h5py

def load_events(f, row_start=0, row_end=-1):
    events_h5 = h5py.File(f)["events"]

    t = np.array(events_h5["t"][row_start:row_end])
    x = np.array(events_h5["x"][row_start:row_end])
    y = np.array(events_h5["y"][row_start:row_end])
    p = np.array(events_h5["p"][row_start:row_end])

    events = np.zeros((t.size, 4))
    events[:, 0] = t * 1e-6
    events[:, 1] = x
    events[:, 2] = y
    events[:, 3] = p

    return events

def _searchsorted(v, l):
    id_start = 0
    id_end = l.len()

    while True:
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
    events_h5 = h5py.File(f)["events"]

    id_start = _searchsorted(t_start * 1e+6, events_h5["t"])
    id_end = _searchsorted(t_end * 1e+6, events_h5["t"])

    print("{} events".format(id_end - id_start))

    return load_events(f, id_start, id_end)

def load_poses(f):
    poses = np.loadtxt(f)
    poses[:,0] *= 1e-6

    tmp = np.copy(poses[:,1])
    poses[:,1] = poses[:,2]
    poses[:,2] = tmp

    tmp = np.copy(poses[:, 2])
    poses[:, 2] = poses[:, 3]
    poses[:, 3] = tmp

    return poses
