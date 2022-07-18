import sys

sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

from lava.proc.io.source import RingBuffer
from lava.proc.io.sink import RingBuffer as SinkBuffer

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor

from lava.lib.dnf.connect.reshape_int.process import ReshapeInt
from lava.lib.dnf.connect.reshape_bool.process import ReshapeBool

from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from semd.proc.tde.process import TDE
from semd.proc.reshape_conv.process import ReshapeConv
from semd.proc.change_dim.process import ChangeDim

from utils import gen_weights

events_range = np.loadtxt("../../data_processed/events_refact.txt")
# this data is already filtered by time, no need to redo it here
events = events_range

# patch_center = (40, 76)
# patch_size = 150
# x_lim = (patch_center[1] - int(patch_size / 2), patch_center[1] + int(patch_size / 2) - 1)
# y_lim = (patch_center[0] - int(patch_size / 2), patch_center[0] + int(patch_size / 2) - 1)
# print(x_lim)

# events_filtered = events_range
# events_filtered = events_filtered[
#     np.logical_and((x_lim[0] <= events_filtered[:, 1]), (events_filtered[:, 1] <= x_lim[1]))]
# events_filtered = events_filtered[
#     np.logical_and((y_lim[0] <= events_filtered[:, 2]), (events_filtered[:, 2] <= y_lim[1]))]
# # events_filtered = events_filtered[[(y_lim[0] >= events_range[:,2]) & (events_range[:,2] <= y_lim[1])]
# events_filtered = np.array(events_filtered)
# events = events_filtered

t_start = events[0, 0]
duration = events[-1, 0] - events[0, 0]

# prepare event data for input to process using lava.io.source
timesteps = 100
steps = 1
width = 240
height = 180

result = np.zeros((height, width, timesteps), dtype=int)

for e in events:
    x = int(e[1])
    y = int(e[2])

    #     x = int(e[1]) - patch_center[1] + int(patch_size / 2)
    #     y = int(e[2]) - patch_center[0] + int(patch_size / 2)
    time = int((float(e[0]) - t_start) / duration * timesteps) - 1

    if e[3] == 1:
        continue
        # consider only positive events
    result[y, x, time] = 1.0

# input layer
input_n = RingBuffer(result)

shape = (height, width)
conv_shape = (3, 3)
conv_stride = (3, 3)
internal_shape = (shape[0] * shape[1],)
out_shape = (int(shape[0] / conv_shape[0]), int(shape[1] / conv_shape[1]))

conv = ReshapeConv(input_shape=shape, conv_shape=conv_shape, conv_stride=conv_stride)
input_n.s_out.connect(conv.s_in)

lif = LIF(shape=out_shape, du=1.0, dv=0.0, vth=10.0)
conv.s_out.connect(lif.a_in)

detector_shape = (out_shape[0], out_shape[1] - 1)
left_detector = TDE(shape=detector_shape, du=(0.05))

change_dim_excit = ChangeDim(shape=(out_shape), col_del=[1], row_del=[])
change_dim_trig = ChangeDim(shape=(out_shape), col_del=[-1], row_del=[])

lif.s_out.connect(change_dim_excit.s_in)
lif.s_out.connect(change_dim_trig.s_in)

change_dim_trig.s_out.connect(left_detector.t_in)
change_dim_excit.s_out.connect(left_detector.a_in)

output_n = SinkBuffer(shape=detector_shape, buffer=steps)
left_detector.s_out.connect(output_n.a_in)

output_lif = SinkBuffer(shape=out_shape, buffer=steps)
lif.s_out.connect(output_lif.a_in)

output_n_in = SinkBuffer(shape=shape, buffer=steps)
input_n.s_out.connect(output_n_in.a_in)

monitor = Monitor()
monitor.probe(left_detector.u, num_steps=steps)

rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=False)

for i in range(steps):
    run_condition = RunSteps(num_steps=1)
    input_n.run(condition=run_condition, run_cfg=rcfg)

data = output_n.data.get()
data_in = output_n_in.data.get()
data_lif = output_lif.data.get()
data_u = monitor.get_data()
input_n.stop()