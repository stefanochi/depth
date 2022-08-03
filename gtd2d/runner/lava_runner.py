import sys

sys.path.append("C:/Users/schiavaz/lava-nc/depth/")

from .runner import Runner
from . import flow_utils
import numpy as np
from tqdm import tqdm
import scipy.sparse

from lava.proc.io.source import RingBuffer
from lava.proc.io.sink import RingBuffer as SinkBuffer
from lava.proc.monitor.process import Monitor

from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from semd.proc.semd_2d.process import Semd2dLayer
from semd.proc.camera_input.process import CameraInputLayer
from semd.proc.float_input.process import RingBuffer as FloatInput
from semd.proc.events_sink.process import EventsSink


class LavaRunner(Runner):
    def __init__(self, events, cam_poses, shape, camera_calib, timesteps, imu_data, cfg):
        self.events = events
        self.cam_poses = cam_poses
        self.shape = tuple(shape)
        self.camera_calib = camera_calib
        self.timesteps = int(timesteps)

        self.cfg = cfg
        self.conv_stride = tuple(cfg["conv_stride"])
        self.conv_shape = tuple(cfg["conv_shape"])
        self.thresh_conv = cfg["thresh_conv"]  # threshold for the subsampling layer
        self.avg_thresh = cfg["avg_thresh"]
        self.avg_shape = tuple(cfg["avg_shape"])
        self.avg_min_meas = cfg["avg_min_meas"]
        self.avg_alpha = cfg["avg_alpha"]
        self.floating = cfg["lava_floating"]
        self.debug = cfg["debug_output"]
        self.imu = cfg["use_imu"]
        self.imu_data = imu_data

        self.input_buffer = self.gen_input_data(self.events, self.shape, self.timesteps)
        self.vel_data = self.imu_data if self.imu else self.cam_poses
        self.vel_input_buffer = self.gen_cam_input_data(self.events, self.vel_data, self.timesteps)
        return

    def gen_input_data(self, events, shape, timesteps):
        t_start = events[0, 0]
        duration = events[-1, 0] - events[0, 0]

        events_buffer = np.zeros((shape[0], shape[1], timesteps), dtype=int)
        for e in events:
            x = int(e[1])
            y = int(e[2])

            time = int((float(e[0]) - t_start) / duration * timesteps) - 1

            pol = 0
            if e[3] == 1:
                pol = 1
            if e[3] == 0:
                pol = -1

            events_buffer[y, x, time] = pol
        return events_buffer

    def gen_cam_input_data(self, events, data, timesteps):

        t_start = events[0, 0]
        duration = events[-1, 0] - events[0, 0]

        velocities_buffer = np.zeros((3, timesteps))

        for i in range(timesteps):
            curr_time = t_start + (duration / timesteps) * i

            vel = flow_utils.vel_at_time(data, curr_time, self.imu)
            velocities_buffer[:, i] = vel[1:4]

        return velocities_buffer

    def run(self):

        semd = Semd2dLayer(shape=self.shape,
                           conv_shape=self.conv_shape,
                           conv_stride=self.conv_stride,
                           thresh_conv=self.thresh_conv,
                           avg_thresh=self.avg_thresh,
                           avg_conv_shape=self.avg_shape,
                           avg_alpha=self.avg_alpha)

        cam_input = CameraInputLayer(shape=self.shape,
                                     focal_length=self.camera_calib[0],
                                     center_x=self.camera_calib[2],
                                     center_y=self.camera_calib[3])

        out_shape = semd.out_shape

        input_n = RingBuffer(self.input_buffer)
        input_cam = FloatInput(self.vel_input_buffer)

        output_u = EventsSink(shape=out_shape)
        output_v = EventsSink(shape=out_shape)
        # output_d = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        # output_avg = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        cam_output_x = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        cam_output_y = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        raw_depth_sink = EventsSink(shape=out_shape)
        mean_depth_sink = EventsSink(shape=out_shape)
        # DEBUG
        debug_output = EventsSink(shape=out_shape)
        avg_debug_output = EventsSink(shape=out_shape)

        input_n.s_out.connect(semd.s_in)
        input_cam.s_out.connect(cam_input.s_in)
        cam_input.x_out.connect(cam_output_x.a_in)
        cam_input.y_out.connect(cam_output_y.a_in)

        semd.u_out.connect(output_u.a_in)
        semd.v_out.connect(output_v.a_in)
        # semd.d_out.connect(output_d.a_in)

        semd.d_out.connect(raw_depth_sink.a_in)
        semd.avg_out.connect(mean_depth_sink.a_in)

        # semd.avg_out.connect(output_avg.a_in)
        cam_input.x_out.connect(semd.tu_in)
        cam_input.y_out.connect(semd.tv_in)
        # DEBUG
        semd.debug_out.connect(debug_output.a_in)
        semd.avg_debug.connect(avg_debug_output.a_in)
        print("total steps: {}".format(self.timesteps))

        rcnd = RunSteps(num_steps=1)
        if self.floating:
            rcfg = LifRunConfig(select_tag='floating_pt')
        else:
            rcfg = LifRunConfig(select_tag='fixed_pt')
        times = np.linspace(self.events[0, 0], self.events[-1, 0], self.timesteps)
        step_t = (times[-1] - times[0]) / self.timesteps

        raw_depth_sparse = []
        mean_depth_sparse = []
        u_sparse = []
        v_sparse = []
        samples_sparse = []
        mean_debug_sparse = []

        for t in tqdm(range(int(self.timesteps))):
            semd.run(condition=rcnd, run_cfg=rcfg)
            data_raw = raw_depth_sink.events_data.get()
            data_mean = mean_depth_sink.events_data.get()

            data_u = output_u.events_data.get()
            data_v = output_v.events_data.get()

            sparse_matrix_raw = scipy.sparse.csr_matrix(data_raw) * step_t
            sparse_matrix_mean = scipy.sparse.csr_matrix(data_mean) * step_t

            sparse_matrix_u = scipy.sparse.csr_matrix(data_u) * step_t
            sparse_matrix_v = scipy.sparse.csr_matrix(data_v) * step_t

            if self.debug:
                data_samples_debug = debug_output.events_data.get()
                data_mean_debug = avg_debug_output.events_data.get()
                sparse_samples_debug = scipy.sparse.csr_matrix(data_samples_debug) * step_t
                sparse_mean_debug = scipy.sparse.csr_matrix(data_mean_debug) * step_t
                samples_sparse.append(sparse_samples_debug)
                mean_debug_sparse.append(sparse_mean_debug)

            raw_depth_sparse.append(sparse_matrix_raw)
            mean_depth_sparse.append(sparse_matrix_mean)
            u_sparse.append(sparse_matrix_u)
            v_sparse.append(sparse_matrix_v)

        cam_x_data = cam_output_x.data.get()
        cam_y_data = cam_output_y.data.get()

        input_n.stop()

        output = {
            "times": times,
            "raw_depths": raw_depth_sparse,
            "mean_depths": mean_depth_sparse,
            "cam_poses": self.cam_poses,
            "cam_calib": self.camera_calib,
            "cfg": self.cfg,
            "flow_u": u_sparse,
            "flow_v": v_sparse,
            "cam_x": cam_x_data,
            "cam_y": cam_y_data,
            "samples": samples_sparse,
            "mean_debug": mean_debug_sparse,
            "imu_data": self.imu
        }

        return output


class LifRunConfig(RunConfig):
    """Run configuration selects appropriate LIF ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""

    def __init__(self, custom_sync_domains=None, select_tag='fixed_pt'):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag

    def select(self, proc, proc_models):
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError("No legal ProcessModel found: {}".format(pm))
