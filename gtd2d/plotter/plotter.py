import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import open3d as o3d
import matplotlib
import matplotlib.cm as cm
import scipy.sparse


class Plotter:
    """Class the handles all the processing of the output data
    to produce plots and data. The putput format of the runners will be different,
    but the data and information return should be the same and comparable."""

    def __init__(self, out, base_path="./"):
        """
        Takes a dict with the frames containing the eaw depths measurements and the filtered ones
        additionally.
            raw depths: the raw depth measurements
            mean depths: the mean filtered measurements
            median depths: the result of the median filter (only for python version)
            cam_poses: the history of camera poses
            cam_calib: the intrinsic parameters of the camera
        :param out: the output of the runner
        """
        self.output = out
        self.times = out["times"]
        # convert sparse to result to dense matrices
        self.raw_depths_sparse = out["raw_depths"]
        self.mean_depths_sparse = out["mean_depths"]
        if out["cfg"]["use_lava"]:
            self.raw_depths = np.array([s.toarray() for s in self.raw_depths_sparse])
            self.mean_depths = np.array([s.toarray() for s in self.mean_depths_sparse])
        else:
            self.raw_depths = self.raw_depths_sparse
            self.mean_depths = self.mean_depths_sparse
            self.median_depths = out["median_depths"]
        self.cam_poses = out["cam_poses"]
        self.cam_calib = out["cam_calib"]
        self.shape = out["cfg"]["dvs_shape"]
        self.subsampling = out["cfg"]["subsampling_factor"]
        self.path = base_path
        self.load_groundtruth()

    def load_groundtruth(self):
        self.gt_depths = np.load(self.path + "gt_depths.npy")
        gt_times = np.genfromtxt(self.path + "depthmaps.txt", dtype="str")
        self.gt_times = gt_times[:, 0].astype(np.float64)

    def get_frame(self, result_type, start, end=None):
        """
        Returns a frame of the results in the specified ids range.
        The range can be either a single id or a range defined by start and end.
        The result has to be normalized by the number of measurements for each point. Important for
        longer ranges of frames
        :param result_type: either raw, mean or median
        :param start: the starting id
        :param end: the end id (optional)
        :return: a matrix representing the frame
        """
        if result_type == "raw":
            depths = self.raw_depths
        elif result_type == "mean":
            depths = self.mean_depths
        else:
            raise Exception("Only mean and raw implemented")

        if end is None:
            end = start + 1
        depths_sum = np.nansum(depths[start:end], axis=0)
        n_measurements = np.nansum(
            np.logical_and(
                depths[start:end] != np.nan,
                depths[start:end] != 0), axis=0)

        normalized_meas = np.divide(depths_sum, n_measurements, where=depths_sum != 0.0, out=np.zeros_like(depths_sum))
        return normalized_meas

    def _image2pointcloud(self, d, v_range=None):
        """
        Transform a single frame to a 3d point cloud using the pixel values as the depth.
        :param d: the frame from which to generate the point-cloud
        :param v_range: the depth range to consider. Points outside this range will be ignored
        :return: the point-cloud corresponding to the frame
        """
        if not self.output["cfg"]["use_lava"]:
            # the input is a dense matrix
            points = []
            for x in range(d.shape[1]):
                for y in range(d.shape[0]):
                    if np.isnan(d[y, x]):
                        continue
                    if v_range is not None:
                        if v_range[0] > d[y, x] or d[y, x] > v_range[1]:
                            continue
                    points.append([x, y, d[y, x]])
            return np.array(points)
        else:
            # the input is a sparse matrix
            i, j, v = scipy.sparse.find(d)
            points = np.zeros((i.shape[0], 3))
            points[:, 0] = j
            points[:, 1] = i
            points[:, 2] = v
            points = points[
                np.logical_and(v >= v_range[0], points[:, 2] < v_range[1])]
            return points

    def project3d(self, points, calib, pose):
        """
        Given a point cloud and the camera position, transform all point to the corresponding world position
        :param points: the point-cloud to be transformed
        :param calib: the cameras intrinsic parameters
        :param pose: the camera extrinsic parameters
        :return: the point-cloud in world coordinates
        """
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

    def gen_world_pointcloud(self, result_type, v_range):
        """
        Generate the point-cloud in world coordinates
        :param result_type: either "raw" or "mean"
        :return: the point-cloud
        """
        if result_type == "raw":
            depths = self.raw_depths_sparse
        elif result_type == "mean":
            depths = self.mean_depths_sparse
        else:
            raise Exception("Only mean and raw implemented")

        projected_points = np.zeros((1, 4))
        ps = []
        p_init = self.cam_poses[0]
        for i, d in enumerate(tqdm(depths)):
            points = self._image2pointcloud(d, v_range)

            idx = np.searchsorted(self.cam_poses[:, 0], self.times[i])
            p0 = self.cam_poses[idx - 1]
            p1 = self.cam_poses[idx]
            y = (self.times[i] - p0[0]) / (p1[0] - p0[0])
            p = (p0 * (1 - y) + p1 * (y))  # interpolate the camera position
            p[1:4] = p[1:4].reshape(1, 3) @ np.diag([-1, 1, 1])  # TODO why the -1?

            ps.append([p[2], p[3], p[1]])

            proj = self.project3d(points, self.cam_calib, p)
            if proj.size == 0:
                continue
            projected_points = np.concatenate((projected_points, proj))

        return projected_points

    def points_to_open3d_pointcloud(self, points, cmap="jet", z=2):
        """
        Transform a point-cloud to open3d format
        :param points: input point-cloud
        :param cmap: colormap
        :param z: which axis is the z (from which to take the color)
        :return: the open3d point-cloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        minima = min(points[:, z])
        maxima = max(points[:, z])

        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = mapper.to_rgba(points[:, z])[:, :3]
        print(colors.shape)

        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def plot_open3d(self, result_type, z=2, v_range=None):
        points = self.gen_world_pointcloud(result_type, v_range)
        print(points[:,z].min())
        print(points[:,z].max())
        points_o3d = self.points_to_open3d_pointcloud(points[:, :3], z=z)
        o3d.visualization.draw_geometries([points_o3d])

    def measure_errors(self, result_type, sum_range=25):
        """
        Measure the error between the measurements and the ground truth data.
        The gt data is only available at certain times. The error is measured only around those times
        :param result_type: either "raw" or "mean"
        :return: the error for each gt depth
        """
        if result_type == "raw":
            depths = self.raw_depths
        elif result_type == "mean":
            depths = self.mean_depths
        else:
            raise Exception("Only mean and raw implemented")
        subsampling = self.subsampling
        shape = self.shape
        times = self.times
        errors = []
        rel_err = []

        for i, t in enumerate(self.gt_times):
            if t > times[-1] or t < times[0]:
                continue
            idx = np.searchsorted(times, float(t))
            s_id = max(0, idx - sum_range)
            e_id = min(depths.shape[0], idx + sum_range)
            m_depth = self.get_frame(result_type, start=s_id, end=e_id)

            gt_d = self.gt_depths[i, ::subsampling, ::subsampling]

            diff_r = np.subtract(m_depth, gt_d, where=m_depth != 0.0, out=np.zeros_like(m_depth))
            diff = np.abs(diff_r)

            diff_rel = np.divide(diff, gt_d, where=diff != 0.0, out=np.zeros_like(diff))

            diff[np.isclose(diff, 0.0)] = np.nan
            errors.append(diff)

            diff_rel[np.isclose(diff_rel, 0.0)] = np.nan
            rel_err.append(diff_rel)

        return np.array(errors)
