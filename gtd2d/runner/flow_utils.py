import numpy as np
from scipy.spatial.transform import Rotation as R


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
                [(xi * yi) / f, -(xi ** 2) / f - f, yi],
                [(yi ** 2) / f + f, -(yi * xi) / f, -xi]
            ])
            r = m @ w

            u_flow[y, x] = r[0]
            v_flow[y, x] = r[1]

    return u_flow, v_flow


def vel_at_time(poses, time):
    """Get the camera velocity at the specified time"""
    idx = np.searchsorted(poses[:, 0], time)

    pose_start = poses[idx + 1]
    pose_end = poses[idx]

    # translational position difference
    t1 = pose_start[1:4]
    t2 = pose_end[1:4]
    t_vel = (t2 - t1) / (pose_end[0] - pose_start[0])

    # angular position difference
    ang1 = R.from_quat(pose_start[4:]).as_euler("xyz")
    ang2 = R.from_quat(pose_end[4:]).as_euler("xyz")
    a_vel = (ang2 - ang1) / (pose_end[0] - pose_start[0])

    return np.array([time, t_vel[0], t_vel[1], t_vel[2], a_vel[0], a_vel[1], a_vel[2]])
