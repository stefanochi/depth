import numpy as np

def filter_conv(events, shape, factor=5, thresh=1.0/3.0):
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

        conv_acc[y, x] += 1 if e[3] == 1 else -1

        if np.abs(conv_acc[y, x]) >= conv_shape[0] * conv_shape[1] * thresh:
            event_conv.append([e[0], x, y, e[3]])
            conv_acc[y, x] = 0

    event_conv = np.array(event_conv)
    return event_conv, result_shape

def filter_conv2(events, shape, factor=5, thresh=1.0/3.0):
    conv_shape = (factor, factor)
    conv_stride = (1, 1)
    result_shape = (int(shape[0] / conv_stride[0]), int(shape[1] / conv_stride[1]))
    conv_acc = np.zeros(result_shape)
    event_conv = []
    hf = int(factor / 2)
    print(hf)
    print(conv_shape[0] * conv_shape[1] * thresh)
    for e in events:
        # if e[3] != 1:
        #     continue

        x = int(e[1] / conv_stride[0])
        y = int(e[2] / conv_stride[1])


        for xi in range(-hf, hf+1):
            for yi in range(-hf, hf+1):
                if y + yi < 0 or y+yi>=result_shape[1]:
                    continue
                if x + xi < 0 or x + xi >= result_shape[0]:
                    continue
                conv_acc[y+yi, x+xi] += 1 if e[3] == 1 else -1

                if np.abs(conv_acc[y, x]) > conv_shape[0] * conv_shape[1] * thresh:
                    event_conv.append([e[0], x+xi, y+yi, e[3]])
                    conv_acc[y+yi, x+xi] = 0

    event_conv = np.array(event_conv)
    return event_conv, result_shape

def filter_time(events, t_start, t_end):
    id_start = np.searchsorted(events[:,0], t_start)
    id_end = np.searchsorted(events[:,0], t_end)

    return events[id_start:id_end, :]


def filter_patch(events, patch_center, patch_size):
    x_lim = (patch_center[1] - int(patch_size / 2), patch_center[1] + int(patch_size / 2) - 1)
    y_lim = (patch_center[0] - int(patch_size / 2), patch_center[0] + int(patch_size / 2) - 1)

    events_filtered = np.copy(events)
    events_filtered = events_filtered[
        np.logical_and((x_lim[0] <= events_filtered[:, 1]), (events_filtered[:, 1] < x_lim[1]))]
    events_filtered = events_filtered[
        np.logical_and((y_lim[0] <= events_filtered[:, 2]), (events_filtered[:, 2] < y_lim[1]))]
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

def patch_variance(img, dist):
    result = np.zeros(img.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img.mask[y, x]:
                continue
            area = img[y - dist:y + dist + 1, x - dist:x + dist + 1].compressed()
            if len(area) == 0:
                continue
            result[y, x] = np.var(area)
    return result

def median_filter(img, dist):
    result = np.zeros(img.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img.mask[y, x]:
                continue
            area = img[y-dist:y+dist+1,x-dist:x+dist+1].filled(0)
            l = []
            for a in area.flatten():
                if a == 0:
                    continue
                l.append(a)
            l = np.array(l)
            result[y, x] = np.median(l)
    return result

def mean_filter(img, dist):
    result = np.zeros(img.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img.mask[y, x]:
                continue
            area = img[y-dist:y+dist+1,x-dist:x+dist+1].filled(0)
            l = []
            for a in area.flatten():
                if a == 0:
                    continue
                l.append(a)
            l = np.array(l)
            result[y, x] = np.mean(l)
    return result

def radius_filter(img, dist, thresh):
    result = np.zeros(img.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img.mask[y, x]:
                continue
            area = img[y-dist:y+dist+1,x-dist:x+dist+1]
            n = area.size - area.mask.astype(int).sum()
            #print(n)
            if n >= thresh:
                result[y, x] = img.data[y, x]
    return result