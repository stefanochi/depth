import numpy as np
from scipy.spatial.transform import Rotation as R

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

def median_from_dict(time_diff_dict, warped_shape, threshold):
    warped_img = np.zeros(warped_shape)
    warped_count = np.zeros(warped_shape)
    for k in time_diff_dict:
        y = k[0]
        x = k[1]
        if len(time_diff_dict[k]) < threshold:
            continue
        warped_img[y, x] = np.median(time_diff_dict[k])
        warped_count[y, x] = len(time_diff_dict[k])

    return warped_img, warped_count

def mean_from_dict(time_diff_dict, warped_shape, threshold):
    warped_img = np.zeros(warped_shape)
    warped_count = np.zeros(warped_shape)
    for k in time_diff_dict:
        y = k[0]
        x = k[1]
        if len(time_diff_dict[k]) < threshold:
            continue
        warped_img[y, x] = np.mean(time_diff_dict[k])
        warped_count[y, x] = len(time_diff_dict[k])

    return warped_img, warped_count

def compute_time_difference(events, shape, dist=1, check_polarity=True, polarity=0):
    last_time = np.zeros(shape)
    time_difference = np.zeros(shape)
    for e in events:
        if check_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        last_time[y, x] = e[0]

        if x >= shape[1] - dist:
            continue
        if last_time[y, x + dist] != 0:
            time_difference[y, x] = e[0] - last_time[y, x + dist]

    return time_difference

def td2depth_slider(u):
    v = 0.2901460156636141
    f = 335.419462958

    d = np.divide(f, u, where=u > 0.0) * v
    d = np.ma.masked_where(d <= 0, d)
    return d

def v2depth_slider_close(u):
    v = 0.158864
    f = 335.419462958

    d = np.divide(f, u, where=u > 0.0) * v
    d = np.ma.masked_where(d <= 0, d)
    return d

def v2depth(u, v, f):
    d = np.divide(f, u, where=u > 0.0) * v
    d = np.ma.masked_where(d <= 0, d)
    return d

def project3d(points, calib, pose):
    K = np.array([
        [calib[0], 0, calib[2]],
        [0, calib[1], calib[3]],
        [0, 0, 1]
    ])

    t = pose[1:4].reshape(3, 1)
    q = pose[4:]

    r = R.from_quat(q)
    r_m = r.as_matrix()

    K_b = np.block([
        [K, np.zeros((3, 1))],
        [np.zeros((1, 3)), 1]
    ])
    rt_b = np.block([
        [r_m, t],
        [np.zeros((1, 3)), 1]
    ])

    proj = []
    for p in points:
        if p[2] == 0:
            continue
        po = np.array([p[0], p[1], 1, 1 / p[2]])
        test = p[2] * np.linalg.inv(K_b @ rt_b) @ po
        proj.append(test.transpose())

    proj = np.array(proj)
    return proj