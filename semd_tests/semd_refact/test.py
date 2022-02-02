import sys

sys.path.append("../../")

import numpy as np

from lava.proc.io.source import RingBuffer
from lava.proc.io.sink import RingBuffer as SinkBuffer

from lava.proc.conv.process import Conv#
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense

from lava.lib.dnf.connect.reshape_int.process import ReshapeInt
from lava.lib.dnf.connect.reshape_bool.process import ReshapeBool


from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from semd.proc.tde.process import TDE
from semd.proc.reshape_conv.process import ReshapeConv

from utils import gen_weights

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

result = np.zeros((height, width, timesteps), dtype=int)

for e in events:
    #     x = int(e[1])
    #     y = int(e[2])

    x = int(e[1]) - patch_center[1] + int(patch_size / 2)
    y = int(e[2]) - patch_center[0] + int(patch_size / 2)
    time = int((float(e[0]) - t_start) / duration * timesteps) - 1

    # if e[3] == 1:
    #     #consider only positive events
    result[y, x, time] = 1

# input layer
input_n = RingBuffer(result)

shape = (height, width)
conv_factor = 3
internal_shape = (shape[0] * shape[1],)
out_shape = (int(shape[0] / conv_factor), int(shape[1] / conv_factor))

# conv_weight = np.ones((1, conv_factor, conv_factor, 1))
# conv = Conv(
#     input_shape=(shape[0], shape[1], 1),
#     weight=conv_weight,
#     stride=(conv_factor, conv_factor)
#     )
#
# input_n.s_out.connect(conv.s_in)
# conv.a_out.connect(lif.a_in)

conv = ReshapeConv(input_shape=shape, conv_shape=(3,3), conv_stride=(3,3))
lif = LIF(shape=out_shape)
input_n.s_out.connect(conv.s_in)
conv.s_out.connect(lif.a_in)

detector_shape = (out_shape[0], out_shape[1] - 1)
left_detector = TDE(shape=detector_shape, du=(0.05))

excit_w = gen_weights(out_shape, detector_shape, -1, 0).transpose()
trig_w = gen_weights(out_shape, detector_shape, 1, 0).transpose()

print(excit_w.shape)

dense_excit = Dense(shape=(excit_w.shape), weights=excit_w)
dense_trig = Dense(shape=trig_w.shape, weights=trig_w)

reshape_lif = ReshapeBool(shape_in=(out_shape[0], out_shape[1]), shape_out=((out_shape[0]*out_shape[1], )))
lif.s_out.connect(reshape_lif.s_in)

reshape_lif.s_out.connect(dense_excit.s_in)
reshape_lif.s_out.connect(dense_trig.s_in)

reshape_excit = ReshapeInt(shape_in=(detector_shape[0]*detector_shape[1], ), shape_out=(detector_shape))
reshape_trig = ReshapeInt(shape_in=(detector_shape[0]*detector_shape[1], ), shape_out=(detector_shape))

dense_excit.a_out.connect(reshape_excit.s_in)
dense_trig.a_out.connect(reshape_trig.s_in)

reshape_excit.s_out.connect(left_detector.a_in)
reshape_trig.s_out.connect(left_detector.t_in)

output_n = SinkBuffer(shape=detector_shape, buffer=20)
left_detector.s_out.connect(output_n.a_in)

rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=False)

run_condition = RunSteps(num_steps=timesteps)
input_n.run(condition=run_condition, run_cfg=rcfg)
print(output_n.data.get())
input_n.stop()