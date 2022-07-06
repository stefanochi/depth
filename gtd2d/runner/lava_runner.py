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


class LavaRunner(Runner):
    def __init__(self, events, cam_poses, shape, camera_calib, timesteps):
        self.events = events
        self.cam_poses = cam_poses
        self.shape = shape
        self.camera_calib = camera_calib
        self.timesteps = timesteps

        self.conv_stride = (1, 1)
        self.conv_shape = (1, 1)
        self.thresh_conv = 0.5  # threshold for the subsampling layer

        self.tu = int(-186.19655666 / 2)
        self.tv = 0

        self.input_buffer = self.gen_input_data(self.events, self.shape, self.timesteps)
        return

    def gen_input_data(self, events, shape, timesteps):
        t_start = events[0, 0]
        duration = events[-1, 0] - events[0, 0]

        result = np.zeros((shape[0], shape[1], timesteps), dtype=int)
        for e in events:
            x = int(e[1])
            y = int(e[2])

            time = int((float(e[0]) - t_start) / duration * timesteps) - 1

            pol = 0
            if e[3] == 1:
                pol = 1
            if e[3] == 0:
                pol = -1

            result[y, x, time] = pol
        return result

    def run(self):
        shape = self.shape
        conv_stride = self.conv_stride
        conv_shape = self.conv_shape

        semd = Semd2dLayer(shape=self.shape,
                           conv_shape=self.conv_shape,
                           conv_stride=self.conv_stride,
                           thresh_conv=self.thresh_conv,
                           tu=self.tu,
                           tv=self.tv)

        out_shape = semd.out_shape
        detector_shape = semd.detector_shape

        input_n = RingBuffer(self.input_buffer)

        output_u = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        output_v = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        output_d = SinkBuffer(shape=out_shape, buffer=self.timesteps)
        output_avg = SinkBuffer(shape=out_shape, buffer=self.timesteps)

        input_n.s_out.connect(semd.s_in)
        semd.u_out.connect(output_u.a_in)
        semd.v_out.connect(output_v.a_in)
        semd.d_out.connect(output_d.a_in)
        semd.avg_out.connect(output_avg.a_in)

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
