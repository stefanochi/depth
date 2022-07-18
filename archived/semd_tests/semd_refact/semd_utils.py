import sys

sys.path.append("../../")

import numpy as np
from matplotlib import pyplot as plt

from lava.proc.io.source import RingBuffer
from lava.proc.io.sink import RingBuffer as SinkBuffer

from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from semd.proc.semd.process import SemdLayer

from utils import get_events_range

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
    """
    Generate the data for the RingBuffer from the stream of events
    """
    t_start = events[0, 0]
    duration = events[-1, 0] - events[0, 0]
    
    result = np.zeros((shape[0], shape[1], timesteps), dtype=int)
    for e in events:
        x = int(e[1])
        y = int(e[2])

        time = int((float(e[0]) - t_start) / duration * timesteps) - 1

        if e[3] == 1:
            continue

        result[y, x, time] = 1
    return result

def run_sim(args, events, data_steps, sim_steps):
    shape = args["shape"]
    conv_stride = args["conv_stride"]
    conv_shape = args["conv_shape"]
    
    # depends on the network
    semd = SemdLayer(**args)
    
    out_shape = semd.out_shape
    detector_shape = semd.detector_shape
    
    # generate input data 
    input_data = gen_input_data(events, shape, data_steps)
    # create RingBuffer with input data
    input_n = RingBuffer(input_data)
    # connect RingBuffer to network
    input_n.s_out.connect(semd.s_in)
    
    output_n = SinkBuffer(shape=detector_shape, buffer=sim_steps)
    
    semd.s_out.connect(output_n.a_in)

    rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)

    for i in range(sim_steps):
        run_condition = RunSteps(num_steps=1)
        input_n.run(condition=run_condition, run_cfg=rcfg)
    data = output_n.data.get()
    input_n.stop()
    
    return data