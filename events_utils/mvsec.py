import numpy as np
import hdf5plugin
import h5py

def load_events(f, row_start=0, row_end=-1):
    data = h5py.File(f)
    events_h5 = data['davis']['left']['events']

    t = np.array(events_h5[row_start:row_end, 2])
    x = np.array(events_h5[row_start:row_end, 0])
    y = np.array(events_h5[row_start:row_end, 1])
    p = np.array(events_h5[row_start:row_end, 3])

    events = np.zeros((t.size, 4))
    events[:, 0] = t * 1e-6
    events[:, 1] = x
    events[:, 2] = y
    events[:, 3] = p

    return events

def _searchsorted(v, l):
    id_start = 0
    id_end = l.size
    print(id_end)

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
    data = h5py.File(f)
    events_h5 = data['davis']['left']['events']

    print(events_h5[0,2] + t_start * 1e+6)

    id_start = _searchsorted(events_h5[0,2] + t_start * 1e+6, events_h5[:, 2])
    id_end = _searchsorted(events_h5[0,2] + t_end * 1e+6, events_h5[:, 2])


    print(id_start)
    print("{} events".format(id_end - id_start))

    return load_events(f, id_start, id_end)
