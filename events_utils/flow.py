import numpy as np


def get_translational_flow(t, f, C, shape):
    u_flow = np.zeros(shape)
    v_flow = np.zeros(shape)
    for x in range(u_flow.shape[1]):
        for y in range(u_flow.shape[0]):
            # shift coordinates to be centered
            xi = x - C[0]
            yi = np.abs(y - u_flow.shape[0]) - C[1]

            # compute image flow
            m = np.array([
                [-f, 0, xi],
                [0, -f, yi]
            ])
            r = m @ t

            u_flow[y, x] = r[0]  # x flow
            v_flow[y, x] = r[1]  # y flow

    return u_flow, v_flow


def get_angular_flow(w, f, C, shape):
    u_flow = np.zeros(shape)
    v_flow = np.zeros(shape)
    for x in range(u_flow.shape[1]):
        for y in range(u_flow.shape[0]):
            # center
            xi = x - C[0]
            yi = np.abs(y - u_flow.shape[0]) - C[1]

            m = np.array([
                [(xi*yi)/f, -(xi**2)/f - f, yi],
                [(yi**2)/f + f, -(yi*xi)/f, -xi]
            ])
            r = m @ w

            u_flow[y, x] = r[0]
            v_flow[y, x] = r[1]

    return u_flow, v_flow

def depth_from_flow(U, V, t_U, t_V, a_U, a_V):
    shape = U.shape
    test_u = np.zeros(shape)
    test_v = np.zeros(shape)
    test = np.zeros(shape)
    for x in range(U.shape[1]):
        for y in range(U.shape[0]):

            tu = t_U[y, x]
            tv = t_V[y, x]

            au = a_U[y, x]
            av = a_V[y, x]

            u = U[y, x]
            v = V[y, x]

            if u == 0.0 and v == 0.0:
                test[y, x] = np.nan
                continue

            test_u[y, x] = u - au
            test_v[y, x] = v - av

            # a = np.array([tu, tv]).reshape(2, 1)
            # b = np.array([u - au, v - av])
            # d = np.linalg.lstsq(a, b)[0]
            #         d = nnls(a, b)[0]

            n = np.sqrt(u**2 + v**2)
            d = (u*tu + v*tv) / n

            test[y, x] = d  # z_inv2 #- z_inv2

    return test, test_u, test_v
