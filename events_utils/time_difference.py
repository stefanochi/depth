import numpy as np

def compute_time_difference_warped(events, shape, t_start, warped_shape, dist=1, filter_polarity=False, polarity=0):
    last_time = np.zeros(shape)
    time_diff_dict = {}

    for e in events:
        if filter_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        last_time[y, x] = e[0]

        if x >= shape[1] - dist:
            continue

        if last_time[y, x + dist] != 0:
            speed = dist / (e[0] - last_time[y, x + dist])
            warp_pos_x = int(x + (e[0] - t_start) * speed)
            if warp_pos_x >= warped_shape[1]:
                continue
            if (y, warp_pos_x) in time_diff_dict:
                time_diff_dict[y, warp_pos_x].append(speed)
            else:
                time_diff_dict[y, warp_pos_x] = [speed]

    return time_diff_dict

def median_from_dict(time_diff_dict, warped_shape):
    warped_img = np.zeros(warped_shape)
    for k in time_diff_dict:
        y = k[0]
        x = k[1]
        if len(time_diff_dict[k]) <= 1:
            continue
        warped_img[y, x] = np.median(time_diff_dict[k])

    return warped_img

def compute_time_difference(events, shape, dist=1, check_polarity=True, polarity=0):
    last_time = np.zeros(shape)
    time_difference = np.zeros(shape)
    for e in events:
        if check_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        last_time[y, x] = e[0]

        if x >= shape[1] - dist - 1:
            continue
        if last_time[y, x + dist] != 0:
            time_difference[y, x] = e[0] - last_time[y, x + dist]

    return time_difference