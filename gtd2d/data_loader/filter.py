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
    id_start = np.searchsorted(events[:, 0], t_start)
    id_end = np.searchsorted(events[:, 0], t_end)

    return events[id_start:id_end, :]


def filter_conv(events, shape, factor=5, thresh=1.0 / 2.0):
    conv_shape = (factor, factor)
    conv_stride = (factor, factor)
    result_shape = (int(shape[0] / conv_stride[0]), int(shape[1] / conv_stride[1]))
    conv_acc = np.zeros(result_shape)
    event_conv = []
    for e in events:
        # if e[3] == 0:
        #     continue

        x = int(e[1] / conv_stride[0])
        y = int(e[2] / conv_stride[1])

        if e[3] == 1:
            conv_acc[y, x] += 1
        else:
            conv_acc[y, x] -= 1

        if np.abs(conv_acc[y, x]) >= conv_shape[0] * conv_shape[1] * thresh:
            event_conv.append([e[0], x, y, e[3]])
            conv_acc[y, x] = 0

    event_conv = np.array(event_conv)
    return event_conv, result_shape
