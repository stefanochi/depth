import numpy as np

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
            result = data_in[:-dis_y,:-dis_x]
        else:
            result = data_in[-dis_y:, :-dis_x]
    else:
        if dis_y > 0:
            result = data_in[:-dis_y,-dis_x:]
        else:
            result = data_in[-dis_y:,-dis_x:]

    return result

def get_trig_patch(data_in, dis_x, dis_y):

    return get_excit_patch(data_in, -dis_x, -dis_y)
