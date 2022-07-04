import numpy as np

def filter_refract(events, ref_time):
    time_dict = {}
    result = []
    for e in events:
        key = (e[1], e[2])
        if key not in time_dict:
            time_dict[key] = e
            result.append(e)
            continue

        if time_dict[key][3] == e[3] and e[0] - time_dict[key][0] > ref_time:
            time_dict[key] = e
            result.append(e)

        if time_dict[key][3] != e[3]:
            result.append(e)
            time_dict[key] = e

    return np.array(result)

def filter_time(events, t_start, t_end):
    id_start = np.searchsorted(events[:,0], t_start)
    id_end = np.searchsorted(events[:,0], t_end)

    return events[id_start:id_end, :]