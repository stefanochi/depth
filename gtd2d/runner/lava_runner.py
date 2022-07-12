import sys

sys.path.append("C:/Users/schiavaz/lava-nc/depth/")

from .runner import Runner
from . import flow_utils
import numpy as np
from tqdm import tqdm

from lava.proc.io.source import RingBuffer
from lava.proc.io.sink import RingBuffer as SinkBuffer
from lava.proc.monitor.process import Monitor

from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from semd.proc.semd_2d.process import Semd2dLayer
from semd.proc.camera_input.process import CameraInputLayer


class LavaRunner(Runner):
    def __init__(self, events, cam_poses, shape, camera_calib, timesteps, cfg):
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

        self.input_buffer = self.gen_input_data(self.events, self.shape, self.timesteps)
        self.vel_input_buffer = self.gen_cam_input_data(self.events, self.cam_poses, self.timesteps)
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

    def gen_cam_input_data(self, events, poses, timesteps):

        t_start = events[0, 0]
        duration = events[-1, 0] - events[0, 0]

        velocities_buffer = np.zeros((3, timesteps))

        for i in range(timesteps):
            curr_time = t_start + (duration / timesteps) * i

            vel = flow_utils.vel_at_time(poses, curr_time)
            velocities_buffer[:, i] = vel[1:4]

        return velocities_buffer

    def run(self):

        semd = Semd2dLayer(shape=self.shape,
                           conv_shape=self.conv_shape,
                           conv_stride=self.conv_stride,
                           thresh_conv=self.thresh_conv,
                           avg_thresh=self.avg_thresh,
                           avg_conv_shape=self.avg_shape)

        cam_input = CameraInputLayer(shape=self.shape,
                                     focal_length=self.camera_calib[0],
                                     center_x=self.camera_calib[2],
                                     center_y=self.camera_calib[3])

        out_shape = semd.out_shape

        input_n = RingBuffer(self.input_buffer)
        input_cam = RingBuffer(self.vel_input_buffer)

        output_u = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        output_v = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        output_d = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        output_avg = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        cam_output_x = SinkBuffer(shape=out_shape, buffer=self.timesteps)

        input_n.s_out.connect(semd.s_in)
        input_cam.s_out.connect(cam_input.s_in)

        semd.u_out.connect(output_u.a_in)
        semd.v_out.connect(output_v.a_in)
        semd.d_out.connect(output_d.a_in)
        semd.avg_out.connect(output_avg.a_in)
        cam_input.x_out.connect(semd.tu_in)
        cam_input.y_out.connect(semd.tv_in)

        rcnd = RunSteps(num_steps=self.timesteps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        semd.run(condition=rcnd, run_cfg=rcfg)

        data_u = output_u.data.get()
        data_v = output_v.data.get()
        data_d = output_d.data.get()
        data_avg = output_avg.data.get()

        input_n.stop()

        times = np.linspace(self.events[0, 0], self.events[-1, 0], self.timesteps)
        step_t = (times[-1] - times[0]) / self.timesteps

        output = {
            "times": times,
            "raw_depths": np.moveaxis(data_d, [2, 1, 0], [-3, -1, -2]) * step_t,
            "mean_depths": np.moveaxis(data_avg, [2, 1, 0], [-3, -1, -2]) * step_t,
            "median_depths": None,
            "flow_u": np.moveaxis(data_u, [2, 1, 0], [-3, -1, -2]),
            "flow_v": np.moveaxis(data_v, [2, 1, 0], [-3, -1, -2])
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
        raise AssertionError("No legal ProcessModel found.")
