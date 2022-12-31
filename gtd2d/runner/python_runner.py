from .runner import Runner
from . import flow_utils
import numpy as np
from tqdm import tqdm
import scipy.sparse


class PythonRunner(Runner):
    """Parameters for the execution:
        y"""

    def __init__(self, events, cam_poses, shape, chunk_size, camera_calib, imu_vel, cfg):
        self.events = events
        self.cam_poses = cam_poses
        self.shape = shape
        self.chunk_size = chunk_size
        self.camera_calib = camera_calib
        self.cfg = cfg

        self.last_time = np.full(shape, -1.0)
        self.event_sign = np.full(shape, -1)

        self.filter_size = cfg["avg_shape"][0]
        self.mean_thresh = cfg["avg_thresh"]
        self.filter_time_dim = 0.05
        self.imu = cfg["use_imu"]
        self.dist = cfg["dist"]
        # self.camera_calib[0] /=
        self.imu_vel = imu_vel
        if self.imu:
            self.vel_data = imu_vel
        else:
            self.vel_data = cam_poses
        return

    def run(self):
        n_chunks = np.ceil(self.events.shape[0] / self.chunk_size).astype(int)
        #raw_depths = np.zeros((n_chunks, self.shape[0], self.shape[1]))
        raw_depths = []
        mean_depths = []
        median_depths = []
        #mean_depths = np.zeros((n_chunks, self.shape[0], self.shape[1]))
        #median_depths = np.zeros((n_chunks, self.shape[0], self.shape[1]))
        flow_u = np.zeros((n_chunks, self.shape[0], self.shape[1]))
        flow_v = np.zeros((n_chunks, self.shape[0], self.shape[1]))
        cam_x = np.zeros((n_chunks, self.shape[0], self.shape[1]))
        cam_y = np.zeros((n_chunks, self.shape[0], self.shape[1]))
        times = np.zeros((n_chunks,))

        for i, events_chunk in enumerate(tqdm(np.array_split(self.events, n_chunks, axis=0))):
            # the chunk time used to retrieve the camera velocity is the average time
            # of the chunk
            chunk_time = (events_chunk[0, 0] + events_chunk[-1, 0]) / 2
            times[i] = chunk_time

            # compute the time difference for all the elements in the chunk
            # the time surface is maintained between chunks
            td = self.compute_td(events_chunk, self.shape, dist=self.dist)
            flow_u[i] = td[0]
            flow_v[i] = td[1]

            cam_vel = flow_utils.vel_at_time(self.vel_data, chunk_time, self.imu)
            at = flow_utils.get_translational_flow(cam_vel[1:4],
                                                   self.camera_calib[0],
                                                   [self.camera_calib[2], self.camera_calib[3]],
                                                   self.shape)
            # bw = flow_utils.get_angular_flow(cam_vel[4:],
            #                                  self.camera_calib[0],
            #                                  [self.camera_calib[2], self.camera_calib[3]],
            #                                  self.shape)
            cam_x[i] = at[0] #+ bw[0]
            cam_y[i] = at[1] #+ bw[1]

            raw_depth = self.compute_depth(td, at)
            # raw_depth = raw_depth / (1 - (td[0]*bw[0] + td[1]*bw[1]))

            raw_depth[np.isnan(raw_depth)] = 0
            raw_depths.append(scipy.sparse.csr_matrix(raw_depth))
            # the median and mean should be calculated with the data from the preceding time t
            # find the index of the raw depth corresponding to the lower bound
            # idx = np.abs(times - (chunk_time - self.filter_time_dim)).argmin()
            # # get the sum of the previous measured raw depth, and normalize by the number of measurements if there is
            # # more than one
            # depth_considered_sparse = raw_depths[idx:i + 1]
            # depth_considered = np.zeros((len(depth_considered_sparse), self.shape[0], self.shape[1]))
            # for i, d in enumerate(depth_considered_sparse):
            #     depth_considered[i] = d.toarray()
            # depth_considered[depth_considered == 0.0] = np.nan
            # n_measurements = np.nansum(np.logical_and(depth_considered != 0.0,
            #                                           ~np.isnan(depth_considered)), axis=0)
            # filter_data = np.divide(np.nansum(depth_considered, axis=0), n_measurements,
            #                         where=np.sum(depth_considered, axis=0) != np.nan)
            # filter_data[filter_data == 0] = np.nan
            # # apply both the median filter and the mean filter and add them to the resulting array
            # median_filtered_depth = self.median_filter(filter_data, self.filter_size)
            # mean_filtered_depth = self.mean_filter(filter_data, self.filter_size, self.mean_thresh)
            # mean_depths.append(scipy.sparse.csr_matrix(mean_filtered_depth))
            # median_depths.append(scipy.sparse.csr_matrix(median_filtered_depth))

        output = {
            "times": times,
            "raw_depths": raw_depths,
            #"mean_depths": mean_depths,
            #"median_depths": median_depths,
            "flow_u": flow_u,
            "flow_v": flow_v,
            "cfg": self.cfg,
            "cam_poses": self.cam_poses,
            "cam_calib": self.camera_calib,
            "cam_x": cam_x,
            "cam_y": cam_y,
            "imu_data": self.imu_vel
        }
        print("average chunk duration: {}".format(
            (self.events[-1, 0] - self.events[0, 0]) / n_chunks))

        return output

    def compute_depth(self, td, at):
        U = td[0]  # the measured x time difference
        V = td[1]  # the measured y time difference
        t_U = at[0]  # the x component of the flow (minus depth) at each pixel location
        t_V = at[1]  # the y component of the flow (minus depth) at each pixel location

        depth_map = U * t_U + V * t_V
        # depth_map[depth_map <= 0.0] = np.nan
        return depth_map

    def compute_td(self, events, shape, dist=1):

        U = np.zeros(shape)
        V = np.zeros(shape)

        for e in events:
            x = int(e[1])
            y = int(e[2])

            self.last_time[y, x] = e[0]
            self.event_sign[y, x] = e[3]

            u_td = (e[0] - self.last_time[y - dist, x]
                    if y - dist >= 0 and self.last_time[y - dist, x] != -1.0 and self.event_sign[y - dist, x] == e[3]
                    else -1.0)
            d_td = (e[0] - self.last_time[y + dist, x]
                    if y + dist < shape[0] and self.last_time[y + dist, x] != -1.0 and self.event_sign[y + dist, x] ==
                       e[3]
                    else -1.0)
            r_td = (e[0] - self.last_time[y, x + dist]
                    if x + dist < shape[1] and self.last_time[y, x + dist] != -1.0 and self.event_sign[y, x + dist] ==
                       e[3]
                    else -1.0)
            l_td = (e[0] - self.last_time[y, x - dist]
                    if x - dist >= 0 and self.last_time[y, x - dist] != -1.0 and self.event_sign[y, x - dist] == e[3]
                    else -1.0)

            if u_td <= 0.0:
                if d_td <= 0.0:
                    v_td = 0.0
                else:
                    v_td = d_td
            else:
                if d_td > 0:
                    v_td = d_td if d_td <= u_td else -u_td
                else:
                    v_td = -u_td

            if r_td <= 0.0:
                if l_td <= 0.0:
                    h_td = 0.0
                else:
                    h_td = l_td
            else:
                if l_td > 0:
                    h_td = l_td if l_td <= r_td else -r_td
                else:
                    h_td = -r_td

            u = h_td / dist if h_td != 0.0 else 0.0
            v = v_td / dist if v_td != 0.0 else 0.0

            if u == 0.0 and v == 0.0:
                continue

            U[y, x] = u
            V[y, x] = v

        return U, V

    def mean_filter(self, raw_depth, patch_size, std_mul):
        d_copy = np.copy(raw_depth)
        m = int(patch_size / 2)
        for x in range(d_copy.shape[1]):
            for y in range(d_copy.shape[0]):
                if not np.isnan(raw_depth[y, x]):
                    patch = d_copy[y - m:y + m + 1, x - m:x + m + 1]
                    s = np.nansum(patch) - raw_depth[y, x]
                    n = np.count_nonzero(~np.isnan(patch))
                    if n == 0:
                        continue
                    mean = s / n
                    std = np.std(patch)
                    if np.abs(mean - raw_depth[y, x]) > std_mul * std and n > 1:
                        d_copy[y, x] = raw_depth[y, x]
        return d_copy

    def median_filter(self, raw_depth, patch_size):
        d_copy = np.copy(raw_depth)
        m = int(patch_size / 2)
        for x in range(d_copy.shape[1]):
            for y in range(d_copy.shape[0]):
                if not np.isnan(raw_depth[y, x]):
                    ymin = np.max([0, y - m])
                    ymax = np.min([y + m + 1, d_copy.shape[0]])
                    xmin = np.max([0, x - m])
                    xmax = np.min([x + m + 1, d_copy.shape[1]])
                    patch = raw_depth[ymin:ymax, xmin:xmax]
                    d_copy[y, x] = np.nanmedian(patch) if not np.isnan(raw_depth[y, x]) else np.nan
        return d_copy
