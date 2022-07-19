import numpy as np
import scipy.sparse as sparse


def get_connection_conv(shape, kernel_size):
    n_elems = shape[0] * shape[1]


def get_connection_left(shape):
    n_elems = shape[0] * shape[1]
    d = np.ones(n_elems - 1)
    d[shape[0] - 1::shape[0]] = 0
    conn = np.diag(d, 1)
    return conn


def get_connection_right(shape):
    n_elems = shape[0] * shape[1]
    d = np.ones(n_elems - 1)
    d[shape[0] - 1::shape[0]] = 0
    conn = np.diag(d, -1)
    return conn


def get_connection_down(shape):
    d = np.ones(shape[0] * (shape[1] - 1))
    conn = np.diag(d, -shape[0])
    return conn


def get_connection_up(shape):
    d = np.ones(shape[0] * (shape[1] - 1))
    conn = np.diag(d, shape[0])
    return conn


def get_conv_right(dist, fixed=False):
    # dist is the distance in pixel between which we want to measure the time difference
    l = 2 * dist + 1
    shape = (l, l)
    if fixed:
        weights = np.zeros(shape, dtype=np.int32)
        weights[dist, -1] = 1
    else:
        weights = np.zeros(shape)
        weights[dist, -1] = 1.0
    return np.expand_dims(weights, axis=(0, -1))


def get_conv_left(dist, fixed=False):
    # dist is the distance in pixel between which we want to measure the time difference
    l = 2 * dist + 1
    shape = (l, l)
    if fixed:
        weights = np.zeros(shape, dtype=np.int32)
        weights[dist, 0] = 1
    else:
        weights = np.zeros(shape)
        weights[dist, 0] = 1.0
    return np.expand_dims(weights, axis=(0, -1))


def get_conv_down(dist, fixed=False):
    # dist is the distance in pixel between which we want to measure the time difference
    l = 2 * dist + 1
    shape = (l, l)
    if fixed:
        weights = np.zeros(shape, dtype=np.int32)
        weights[-1, dist] = 1
    else:
        weights = np.zeros(shape)
        weights[-1, dist] = 1.0
    return np.expand_dims(weights, axis=(0, -1))


def get_conv_up(dist, fixed=False):
    # dist is the distance in pixel between which we want to measure the time difference
    l = 2 * dist + 1
    shape = (l, l)
    if fixed:
        weights = np.zeros(shape, dtype=np.int32)
        weights[0, dist] = 1
    else:
        weights = np.zeros(shape)
        weights[0, dist] = 1.0
    return np.expand_dims(weights, axis=(0, -1))

def get_conv_eye(dist, fixed=False):
    # dist is the distance in pixel between which we want to measure the time difference
    l = 2 * dist + 1
    shape = (l, l)
    if fixed:
        weights = np.zeros(shape, dtype=np.int32)
        weights[dist, dist] = 1
    else:
        weights = np.zeros(shape)
        weights[dist, dist] = 1.0
    return np.expand_dims(weights, axis=(0, -1))

def calc_patch_size(input_shape, dis_x, dis_y):
    """
    This method returns the dimension of the patch, when measuring the time between pixel with the specified distance
    :param input_shape: original shape
    :param dis_x: pixel distance along x
    :param dis_y: pixel distance along y
    :return: shape of the patch
    """
    shape = (input_shape[0] - dis_y, input_shape[1] - dis_x)

    return shape


def get_excit_patch(data_in, dis_x, dis_y):
    patch_shape = calc_patch_size(data_in.shape, np.abs(dis_x), np.abs(dis_y))

    result = np.zeros(patch_shape)

    if dis_x > 0:
        if dis_y > 0:
            result = data_in[:-dis_y, :-dis_x]
        else:
            result = data_in[-dis_y:, :-dis_x]
    else:
        if dis_y > 0:
            result = data_in[:-dis_y, -dis_x:]
        else:
            result = data_in[-dis_y:, -dis_x:]

    return result


def get_trig_patch(data_in, dis_x, dis_y):
    return get_excit_patch(data_in, -dis_x, -dis_y)


def get_connection_left_sparse(shape):
    n_elems = shape[0] * shape[1]
    d = np.ones(n_elems - 1)
    d[shape[0] - 1::shape[0]] = 0
    conn = sparse.dia_array((np.array([d]), [1]), shape=shape)
    return conn


def get_connection_right_sparse(shape):
    n_elems = shape[0] * shape[1]
    d = np.ones(n_elems - 1)
    d[shape[0] - 1::shape[0]] = 0
    conn = sparse.dia_array((np.array([d]), [-1]), shape=shape)
    return conn


def get_connection_down_sparse(shape):
    d = np.ones(shape[0] * (shape[1] - 1))
    conn = sparse.dia_array((np.array([d]), [-shape[0]]), shape=shape)
    return conn


def get_connection_up_sparse(shape):
    d = np.ones(shape[0] * (shape[1] - 1))
    conn = sparse.dia_array((np.array([d]), [shape[0]]), shape=shape)
    return conn


def get_eye_sparse(shape):
    return sparse.eye(shape[0], format="dia")


def dense_to_sparse(m):
    return sparse.dia_array(m)
