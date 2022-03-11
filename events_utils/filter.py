import numpy as np

def filter_time(events, t_start, t_end):
    id_start = np.searchsorted(events[:,0], t_start)
    id_end = np.searchsorted(events[:,0], t_end)

    return events[id_start:id_end, :]


def filter_patch(events, patch_center, patch_size):
    x_lim = (patch_center[1] - int(patch_size / 2), patch_center[1] + int(patch_size / 2) - 1)
    y_lim = (patch_center[0] - int(patch_size / 2), patch_center[0] + int(patch_size / 2) - 1)

    events_filtered = np.copy(events)
    events_filtered = events_filtered[
        np.logical_and((x_lim[0] <= events_filtered[:, 1]), (events_filtered[:, 1] <= x_lim[1]))]
    events_filtered = events_filtered[
        np.logical_and((y_lim[0] <= events_filtered[:, 2]), (events_filtered[:, 2] <= y_lim[1]))]
    # events_filtered = events_filtered[[(y_lim[0] >= events_range[:,2]) & (events_range[:,2] <= y_lim[1])]
    # vents_filtered = np.array(events_filtered)

    events_filtered[:, 1] = (events_filtered[:, 1] - patch_center[1] + int(patch_size / 2)).astype(int)
    events_filtered[:, 2] = (events_filtered[:, 2] - patch_center[0] + int(patch_size / 2)).astype(int)

    return events_filtered

def filter_refract(events, ref_time):
    time_dict = {}
    result = []
    for e in events:
        key = (e[1], e[2])
        if key not in time_dict:
            time_dict[key] = e[0]
            result.append(e)
            continue
        if e[0] - time_dict[key] > ref_time:
            time_dict[key] = e[0]
            result.append(e)
    return np.array(result)