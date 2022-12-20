import numba
from numba import literal_unroll
import numpy as np
MAX_HISTORY_MOTION_PREDICTION = 20

@numba.njit
def pad(x_array): # 1 dim input for now
    n = len(x_array)
    output = np.zeros(shape = MAX_HISTORY_MOTION_PREDICTION)
    for i in range(MAX_HISTORY_MOTION_PREDICTION):
        if i < n:
            output[i] = x_array[i]
        else:
            output[i] = x_array[n-1]
    return output

@numba.njit
def build_agent(tuple_of_tuple):
    n = len(tuple_of_tuple)
    output = np.zeros((n, MAX_HISTORY_MOTION_PREDICTION, 2))

    count = 0
    # for agent in literal_unroll(tuple_of_tuple):
    #     x_pos = agent[0]
    #     y_pos = agent[1]
    #     x_pad = pad(x_pos)
    #     y_pad = pad(y_pos)
    #
    #     output[count, :, 0] = x_pad
    #     output[count, :, 1] = y_pad
    #     count += 1

    for i in range(n):
        x_pos = tuple_of_tuple[i][0]
        y_pos = tuple_of_tuple[i][1]
        x_pad = pad(x_pos)
        y_pad = pad(y_pos)

        output[count, :, 0] = x_pad
        output[count, :, 1] = y_pad
        count += 1

    return output