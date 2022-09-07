import sys
sys.path.append("\..")

from lava.proc.lif.process import LIF
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Weights
from lava.lib.dnf.kernels.kernels import MultiPeakKernel
from lava.lib.dnf.operations.operations import Convolution
from lava.lib.dnf.kernels.kernels import SelectiveKernel
from lava.proc.io.source import RingBuffer

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen
from lava.proc.monitor.process import Monitor
import scipy.sparse


import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':
    img_shape = [9, 12]
    depthsxmeter = 10
    d_range = [0.0, 2.0]

    n_depths = int(d_range[1] - d_range[0]) * depthsxmeter
    shape = (img_shape[0], img_shape[1], n_depths)

    dnf = LIF(shape=shape, du=0.12, dv=0.5, vth=10)

    # kernel = SelectiveKernel(amp_exc=4.0,
    #                          width_exc=20.0,
    #                          global_inh=-3)
    kernel = MultiPeakKernel(amp_exc=12,
                             width_exc=[3, 3, 3],
                             amp_inh=-8,
                             width_inh=[2, 2, 3])
    # connect(dnf.s_out, dnf.a_in, [Convolution(kernel)])
    # connect(input_proc.s_out, dnf.a_in, [Weights(7)])

    gauss_pattern = GaussPattern(shape=shape,
                                 amplitude=10,
                                 mean=[5, 2, 5],
                                 stddev=[2, 2, 5])
    spike_generator = RateCodeSpikeGen(shape=shape)
    gauss_pattern.a_out.connect(spike_generator.a_in)

    connect(spike_generator.s_out, dnf.a_in, [Weights(7)])

    time_steps = 10
    monitor = Monitor()
    monitor.probe(dnf.s_out, time_steps)

    # monitor_v = Monitor()
    # monitor_v.probe(dnf.v, time_steps)

    # monitor_u = Monitor()
    # monitor_u.probe(dnf.u, time_steps)

    # Run the DNF
    dnf.run(condition=RunSteps(num_steps=time_steps),
            run_cfg=Loihi1SimCfg(select_tag='floating_pt'))

    # Get probed data from monitor
    probed_data = monitor.get_data()
    # probed_data_v = monitor_v.get_data()
    # probed_data_u = monitor_u.get_data()

    # Stop the execution after getting the monitor's data
    dnf.stop()