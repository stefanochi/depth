import numpy as np

def compute_time_difference(events, shape, dist=1, check_polarity=True, polarity=0):
    last_time = np.zeros(shape)
    time_difference = np.zeros(shape)
    for e in events:
        if check_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        last_time[y, x] = e[0]

        if x >= shape[1] - dist or :
            continue
        if last_time[y, x + dist] != 0:
            time_difference[y, x] = e[0] - last_time[y, x + dist]