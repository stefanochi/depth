import sys

sys.path.append("../")

import numpy as np
from matplotlib import pyplot as plt

from lava.proc.io.source import RingBuffer
from lava.proc.io.sink import RingBuffer as SinkBuffer
from lava.proc.monitor.process import Monitor

from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from semd.proc.semd_2d.process import Semd2dLayer

from utils import get_events_range

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

def filter_patch(events, patch_center, patch_size):
    x_lim = (patch_center[1] - int(patch_size / 2), patch_center[1] + int(patch_size / 2) - 1)
    y_lim = (patch_center[0] - int(patch_size / 2), patch_center[0] + int(patch_size / 2) - 1)

    events_filtered = np.copy(events)
    events_filtered = events_filtered[
        np.logical_and((x_lim[0] <= events_filtered[:, 1]), (events_filtered[:, 1] <= x_lim[1]))]
    events_filtered = events_filtered[
        np.logical_and((y_lim[0] <= events_filtered[:, 2]), (events_filtered[:, 2] <= y_lim[1]))]
    # events_filtered = events_filtered[[(y_lim[0] >= events_range[:,2]) & (events_range[:,2] <= y_lim[1])]
    #vents_filtered = np.array(events_filtered)

    events_filtered[:,1] = (events_filtered[:,1] - patch_center[1] + int(patch_size / 2)).astype(int)
    events_filtered[:,2] = (events_filtered[:,2] - patch_center[0] + int(patch_size / 2)).astype(int)
    
    return events_filtered

def gen_input_data(events, shape, timesteps):
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

def run_sim(args, events, data_steps, sim_steps):
    shape = args["shape"]
    conv_stride = args["conv_stride"]
    conv_shape = args["conv_shape"]
    
#     out_shape = (int(shape[0] / conv_stride[0]), int(shape[1] / conv_stride[1]))
#     detector_shape = (out_shape[0], out_shape[1] - 1)
    
    semd = Semd2dLayer(**args)
    
    out_shape = semd.out_shape
    detector_shape = semd.detector_shape

    input_data = gen_input_data(events, shape, data_steps)
    input_n = RingBuffer(input_data)

    output_u = SinkBuffer(shape=out_shape, buffer=sim_steps)
    output_v = SinkBuffer(shape=out_shape, buffer=sim_steps)
    output_d = SinkBuffer(shape=out_shape, buffer=sim_steps)

    input_n.s_out.connect(semd.s_in)
    semd.u_out.connect(output_u.a_in)
    semd.v_out.connect(output_v.a_in)
    semd.d_out.connect(output_d.a_in)
    
    # monitor = Monitor()
    # monitor.probe(semd.u, sim_steps)

    rcnd = RunSteps(num_steps=sim_steps)
    rcfg = LifRunConfig(select_tag='floating_pt')
    semd.run(condition=rcnd, run_cfg=rcfg)

    # for i in range(sim_steps):
    #     run_condition = RunSteps(num_steps=1)
    #     input_n.run(condition=run_condition, run_cfg=rcfg)
    data_u = output_u.data.get()
    data_v = output_v.data.get()
    data_d = output_d.data.get()
    
    input_n.stop()
    
    return data_u, data_v, data_d