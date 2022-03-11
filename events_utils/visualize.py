import numpy as np
import matplotlib.pyplot as plt

def events2img(events, img_shape, polarity=0, filter_polarity=True):
    img = np.zeros(img_shape)

    for e in events:
        if filter_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        img[y, x] += 1

    return img


def time_surface(events, img_shape, polarity=0, filter_polarity=True):
    img = np.zeros(img_shape)

    for e in events:
        if filter_polarity and e[3] != polarity:
            continue

        x = int(e[1])
        y = int(e[2])

        img[y, x] = e[0]

    return img