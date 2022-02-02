import sys

sys.path.append("../../")

import numpy as np

from lava.proc.io.source import RingBuffer

from semd.proc.semd.process import SemdLayer

from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

events_range = np.loadtxt("../../data_processed/events_refact.txt")
# this data is already filtered by time, no need to redo it here

patch_center = (40, 76)
patch_size = 15
x_lim = (patch_center[1] - int(patch_size / 2), patch_center[1] + int(patch_size / 2) - 1)
y_lim = (patch_center[0] - int(patch_size / 2), patch_center[0] + int(patch_size / 2) - 1)
print(x_lim)

events_filtered = events_range
events_filtered = events_filtered[
    np.logical_and((x_lim[0] <= events_filtered[:, 1]), (events_filtered[:, 1] <= x_lim[1]))]
events_filtered = events_filtered[
    np.logical_and((y_lim[0] <= events_filtered[:, 2]), (events_filtered[:, 2] <= y_lim[1]))]
# events_filtered = events_filtered[[(y_lim[0] >= events_range[:,2]) & (events_range[:,2] <= y_lim[1])]
events_filtered = np.array(events_filtered)
events = events_filtered

t_start = events[0, 0]
duration = events[-1, 0] - events[0, 0]

# prepare event data for input to process using lava.io.source
timesteps = 100
width = patch_size
height = patch_size

result = np.zeros((height * width, timesteps), dtype=int)

for e in events:
    #     x = int(e[1])
    #     y = int(e[2])

    x = int(e[1]) - patch_center[1] + int(patch_size / 2)
    y = int(e[2]) - patch_center[0] + int(patch_size / 2)
    time = int((float(e[0]) - t_start) / duration * timesteps) - 1

    # if e[3] == 1:
    #     #consider only positive events
    result[y * width + x, time] = 1.7

# input layer
input_n = RingBuffer(result)


semd = SemdLayer(shape=(height, width), conv_factor=3)
input_n.s_out.connect(semd.s_in)

rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)

run_condition = RunSteps(num_steps=2)
semd.run(condition=run_condition, run_cfg=rcfg)
# semd.stop()