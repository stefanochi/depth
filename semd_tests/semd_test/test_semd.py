import sys

sys.path.append("../")
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from change_dim.process import ChangeDim
from lava.proc.lif.process import LIF
from lava.proc.io.source import RingBuffer
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.dense.process import Dense
from lava.lib.dnf.kernels.kernels import MultiPeakKernel
from lava.lib.dnf.kernels.kernels import Kernel
from lava.lib.dnf.operations.operations import Convolution
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import ReduceDims
from lava.lib.dnf.operations.operations import Weights
from lava.proc.conv.process import Conv

from tde.process import TDE
import utils


event_data = np.loadtxt("../data/events.txt")
t_start = 1.5
duration = 0.1

#reduce range of events
events = utils.get_events_range(event_data, t_start, duration)

#prepare event data for input to process using lava.io.source
timesteps = 100
width = 240
height = 180 

result = np.zeros((height, width, 1, timesteps))

for e in events:
    x = int(e[1])
    y = int(e[2])
    time = int((float(e[0]) - t_start) / duration  * timesteps)
    
    # if e[3] == 1:
    #     #consider only positive events
    result[y, x, 0, time] = 1.7

#input layer
input_n = RingBuffer(result)

#subsampling layer
subsampling_factor = 3
sub_decay = 1.0
sub_layer_shape = (178, 238)
#create LIF layer
sub_layer = LIF(shape=sub_layer_shape, du=sub_decay, dv=0.01)

#connect layers with convolution
weight=np.ones((1, 3, 3, 1))
conv = Conv(
    input_shape=(180, 240, 1),
    weight=weight,
    stride=(1, 1)
)
input_n.s_out.connect(conv.s_in)
connect(conv.a_out, sub_layer.a_in, ops=[ReduceDims(reduce_dims=2)])
#connect(conv.a_out, sub_layer.a_in)

#time difference encoder
left_detector = TDE(shape=(sub_layer_shape[0], sub_layer_shape[1]-1), du=(0.05))

change_dim1 = ChangeDim(shape=sub_layer_shape, r=0, w=10)
change_dim2 = ChangeDim(shape=sub_layer_shape, r=-1, w=10)

sub_layer.s_out.connect(change_dim1.s_in)
sub_layer.s_out.connect(change_dim2.s_in)

change_dim1.a_out.connect(left_detector.a_in)
change_dim2.a_out.connect(left_detector.t_in)

#monitor
monitor_s = Monitor()
monitor_s.probe(left_detector.s_out, num_steps=timesteps)
# monitor_t = Monitor()
# monitor_t.probe(left_detector.trig, num_steps=timesteps)
# monitor_u = Monitor()
# monitor_u.probe(left_detector.u, num_steps=timesteps)

run_condition = RunSteps(num_steps=timesteps, blocking=True)
left_detector.run(condition=run_condition, run_cfg=Loihi1SimCfg())

data_s = monitor_s.get_data()
#data_t = monitor_t.get_data()
#data_u = monitor_u.get_data()

input_n.stop()

with open('data_s_graded.npy', 'wb') as f:
    np.save(f, data_s, allow_pickle=True)
# with open('data_u.npy', 'wb') as f:
#     np.save(f, data_u, allow_pickle=True)
# with open('data_t.npy', 'wb') as f:
#     np.save(f, data_t, allow_pickle=True)
