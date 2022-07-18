import numpy as np
import matplotlib.pyplot as plt

thetas = np.deg2rad(np.arange(0.0, 91.0, 1))
width, height = 400, 240
diag_len = int(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

# Cache some reusable values
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)
num_thetas = len(thetas)


def get_events_range(event_data, t_start, duration):
    return event_data[(event_data[:, 0] >= t_start) & (event_data[:, 0] < t_start + duration)]


def get_events_in_row(events, y):
    return events[events[:, 2] == y]


def create_accumulator():
    return np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)


def hough_line(accumulator, x, y):
    # Vote in the hough accumulator
    for t_idx in range(num_thetas):
        # Calculate rho. diag_len is added for a positive index
        rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
        accumulator[rho, t_idx] += 1


def reset_accumulator(accumulator):
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)


def get_interpolated_position(time, positions):
    pos = np.copy(positions)
    id1 = np.nanargmin(np.abs(pos[:, 0] - time))
    if pos[id1, 0] > time:
        id2 = id1 - 1
    else:
        id2 = id1 + 1
    p1 = positions[id1]
    p2 = positions[id2]

    times = np.array([p1[0], time, p2[0]])
    times = (times - times.min()) / (times.max() - times.min())
    factor = times[1]

    inter_position = p1[1] + np.abs((p2[1] - p1[1])) * factor

    return inter_position

def get_interpolated_position_y(time, positions):
    pos = np.copy(positions)
    id1 = np.nanargmin(np.abs(pos[:, 0] - time))
    if pos[id1, 0] > time:
        id2 = id1 - 1
    else:
        id2 = id1 + 1
    p1 = positions[id1]
    p2 = positions[id2]

    times = np.array([p1[0], time, p2[0]])
    times = (times - times.min()) / (times.max() - times.min())
    factor = times[1]

    inter_position = p1[2] + np.abs((p2[2] - p1[2])) * factor

    return inter_position


def pos_to_grid(events, positions, pos):
    p_start = get_interpolated_position(events[0, 0], positions)
    p_end = get_interpolated_position(events[-1, 0], positions)

    return int((pos - p_start) / (p_end - p_start) * width)

def pos_to_grid_lim(pos, positions):
    p_min_id = positions[:,1].argmin()
    p_max_id = positions[:,1].argmax()

    p_start = positions[p_min_id][1]
    p_end = positions[p_max_id][1]

    return int((pos - p_start) / (p_end - p_start) * width)


def process_row_data(events, position, accumulator):
    # iterate the events and add information to accumulator

    for e in events:
        # check if event is positive
        if e[3] == 1 or e[3] == 0:
            # get time of event and interpolate the position
            time = e[0]
            inter_position = get_interpolated_position(time, position)
            grid_position = pos_to_grid(events, position, inter_position)
            hough_line(accumulator, grid_position, np.abs(e[1]))


def threshold_accumulator(accumulator, n_elem):
    # get the points in Hough space that are above the threshold
    accumulator_thresh = np.copy(accumulator)
    thresh = np.sort(accumulator.flatten())[-n_elem]
    accumulator_thresh[accumulator_thresh < thresh] = 0
    return accumulator_thresh


def get_pol_of_positives(accumulator):
    return np.where(accumulator > 0)


def plot_hough_space(accumulator):
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot()
    ax.imshow(accumulator)
    plt.show()


def plot_euclidean(params):
    x = np.linspace(0, width)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    for r, t in zip(params[0], params[1]):
        if sin_t[t] != 0.0:
            y = ((r - diag_len - x * cos_t[t]) / sin_t[t])
            ax.plot(x, y)



    plt.ylim([0, height])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()


def get_depth_information(params):
    # given the parameter of the lines return the position in world coordinate
    # and the slope (depth)

    x = np.linspace(0, width)
    a = np.array([])
    p = np.array([])
    for r, t in zip(params[0], params[1]):
        if sin_t[t] != 0.0 and cos_t[t] != 0.0:
            y = (r - diag_len - x * cos_t[t]) / sin_t[t]
            a = np.append(a, (y[49] - y[0]) / (x[49] - x[0]))
            p = np.append(p, ((width / 2 * sin_t[t]) + (r - diag_len)) / cos_t[t])

    # transform position to int and cleanup
    p = np.round(p).astype(int)

    return p, a


def get_depth_at_pix(pos, params):
    #  rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
    result = []
    for r, t in zip(params[0], params[1]):
        pix = (r - diag_len - (pos * cos_t[t])) / sin_t[t]
        result.append([pix, t])

    return np.array(result)


def get_world_map(params):
    # the x axis is the world position and the y axis is the depths
    result = np.zeros((width, len(thetas)))

    for r, t in zip(params[0], params[1]):
        if t > 0:
            x = int((r - diag_len - 120 * sin_t[t]) / cos_t[t])
            y = int(t)
            if 0 < x < width and 0 < y < len(thetas):
                result[x, y] = 256

    return result


def get_row_depths(params):
    result = np.zeros(width)

    for r, t in zip(params[0], params[1]):
        if t > 0:
            x = int((r - diag_len - 120 * sin_t[t]) / cos_t[t])
            if 0 < x < width:
                result[x] = t

    return result

def get_image_depths (params, pos, events, positions):
    #  rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len

    result = np.zeros(height)

    pos_grid = pos_to_grid(events, positions, pos)

    for r, t in zip(params[0], params[1]):
        if t>0:
            y = int((r - diag_len - pos_grid * cos_t[t]) / sin_t[t])
            if 0 < y < height:
                result[y] = t

    return result