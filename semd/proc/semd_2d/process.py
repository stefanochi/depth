import numpy as np

from lava.proc.lif.process import LIF, TernaryLIF
from lava.proc.dense.process import Dense
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.proc.io.source import RingBuffer

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements, tag, requires
from lava.magma.core.resources import CPU

# from utils import calc_patch_size
from semd.proc.tde_2d.process import TDE2D
from semd.proc.reshape_conv.process import ReshapeConv
from semd.proc.change_dim.process import ChangeDim
from semd.proc.add_dim.process import AddDim
from semd.proc.average_layer.process import AverageLayer
from lava.proc.conv.process import Conv

from lava.lib.dnf.connect.reshape_int.process import ReshapeInt
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Convolution, Weights

import semd.proc.semd_2d.utils as utils


class Semd2dLayer(AbstractProcess):
    """sEMD
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape", (1, 1))
        conv_shape = kwargs.pop("conv_shape", (1, 1))
        conv_stride = kwargs.pop("conv_stride", (1, 1))
        thresh_conv = kwargs.pop("thresh_conv", 0.5)
        vth = kwargs.pop("vth", 10)
        avg_thresh = kwargs.pop("avg_thresh", 50)
        avg_min_meas = kwargs.pop("avg_min_meas", 5)
        avg_conv_shape = kwargs.pop("avg_conv_shape", (5, 5))
        avg_alpha = kwargs.pop("avg_alpha", 0.5)

        bias_weight = vth / conv_shape[0] * conv_shape[1] * thresh_conv
        # convolution layer with reshape proc
        conv = ReshapeConv(input_shape=shape,
                           conv_shape=conv_shape,
                           conv_stride=conv_stride,
                           bias_weight=bias_weight)

        self.out_shape = conv.out_shape

        self.s_in = InPort(shape=shape)
        self.tu_in = InPort(shape=shape)
        self.tv_in = InPort(shape=shape)

        self.u_out = OutPort(shape=self.out_shape)
        self.v_out = OutPort(shape=self.out_shape)
        self.d_out = OutPort(shape=self.out_shape)
        self.avg_out = OutPort(shape=self.out_shape)
        self.vth = Var(shape=(1,), init=10)
        # DEBUG
        self.debug_out = OutPort(shape=self.out_shape)
        self.avg_debug = OutPort(shape=self.out_shape)
        # DEBUG
        self.initialize_weights()
        #self.initialize_conv_weights(avg_conv_shape)

        self.counter = Var(shape=shape, init=0)

    def initialize_weights(self, dist=1):

        up_weights = utils.get_conv_up(dist)
        down_weights = utils.get_conv_down(dist)
        left_weights = utils.get_conv_left(dist)
        right_weights = utils.get_conv_right(dist)
        trig_weights = utils.get_conv_eye(dist)

        conv_weight_shape = up_weights.shape

        self.up_weights = Var(shape=conv_weight_shape, init=up_weights)
        self.down_weights = Var(shape=conv_weight_shape, init=down_weights)
        self.left_weights = Var(shape=conv_weight_shape, init=left_weights)
        self.right_weights = Var(shape=conv_weight_shape, init=right_weights)
        self.trig_weights = Var(shape=conv_weight_shape, init=trig_weights)

    def initialize_conv_weights(self, avg_conv_shape):
        n_elems = self.out_shape[0] * self.out_shape[1]
        conv = Convolution(np.full(avg_conv_shape, 1.0))
        conv.configure(self.out_shape)
        conv_weights = conv._compute_weights() - np.eye(n_elems)
        self.conv_avg_weights = Var(shape=conv_weights.shape, init=conv_weights)


@implements(proc=Semd2dLayer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt', 'fixed_pt')
class Semd2dLayerModel(AbstractSubProcessModel):

    def __init__(self, proc):
        # input layer
        shape = proc.init_args.get("shape", (1, 1))
        conv_shape = proc.init_args.get("conv_shape", (1, 1))
        conv_stride = proc.init_args.get("conv_stride", (1, 1))
        thresh_conv = proc.init_args.get("thresh_conv", 0.5)
        vth = proc.init_args.get("vth", 0.9)
        avg_thresh = proc.init_args.get("avg_thresh", 50)
        avg_min_meas = proc.init_args.get("avg_min_meas", 1)
        avg_alpha = proc.init_args.get("avg_alpha", 0.5)
        avg_conv_shape = proc.init_args.get("avg_conv_shape", (1, 1))

        up_weights = proc.vars.up_weights.get()
        down_weights = proc.vars.down_weights.get()
        left_weights = proc.vars.left_weights.get()
        right_weights = proc.vars.right_weights.get()
        trig_weights = proc.vars.trig_weights.get()
        conv_weights_shape = (1, avg_conv_shape[0], avg_conv_shape[1], 1)

        # TODO implement subsampling in lava

        self.td = TDE2D(shape=shape)
        #
        proc.in_ports.tu_in.connect(self.td.t_u)
        proc.in_ports.tv_in.connect(self.td.t_v)
        # create all the connections for the TDE2D
        shape_conv = (shape[0], shape[1], 1)
        self.conn_up = Conv(
            input_shape=shape_conv,
            weight=up_weights,
            padding=1,
            use_graded_spike=True
        )
        self.conn_down = Conv(
            input_shape=shape_conv,
            weight=down_weights,
            padding=1,
            use_graded_spike=True
        )
        self.conn_left = Conv(
            input_shape=shape_conv,
            weight=left_weights,
            padding=1,
            use_graded_spike=True
        )
        self.conn_right = Conv(
            input_shape=shape_conv,
            weight=right_weights,
            padding=1,
            use_graded_spike=True
        )
        self.conn_trig = Conv(
            input_shape=shape_conv,
            weight=trig_weights,
            padding=1,
            use_graded_spike=True
        )
        # connect to the conv layers
        proc.in_ports.s_in.reshape(shape_conv).connect(self.conn_up.s_in)
        proc.in_ports.s_in.reshape(shape_conv).connect(self.conn_down.s_in)
        proc.in_ports.s_in.reshape(shape_conv).connect(self.conn_left.s_in)
        proc.in_ports.s_in.reshape(shape_conv).connect(self.conn_right.s_in)
        proc.in_ports.s_in.reshape(shape_conv).connect(self.conn_trig.s_in)
        # connect the conv to the TDE2D
        self.conn_up.a_out.reshape(shape).connect(self.td.up_in)
        self.conn_down.a_out.reshape(shape).connect(self.td.down_in)
        self.conn_left.a_out.reshape(shape).connect(self.td.left_in)
        self.conn_right.a_out.reshape(shape).connect(self.td.right_in)
        self.conn_trig.a_out.reshape(shape).connect(self.td.trig_in)

        # ----------------------
        # Average Layer
        self.average_layer = AverageLayer(shape=shape, mean_thr=avg_thresh, min_meas=avg_min_meas, avg_alpha=avg_alpha)

        # connections for the average layer

        # connect the TDE2D to the average layer. This is a simple eye connection
        # pixel in one layer connected to the same pixels in the next one-. (Is the connection layer really necessary)
        self.conn_td2avg = Conv(
            input_shape=shape_conv,
            weight=trig_weights,  # just a matrix with 1 in the middle and rest zero
            padding=1,
            use_graded_spike=True)
        self.td.d_out.reshape(shape_conv).connect(self.conn_td2avg.s_in)
        self.conn_td2avg.a_out.reshape(shape).connect(self.average_layer.trig_in)

        # recurrent connection in the average layer. Implemented as a convolution
        # each pixel receives the depth values from the neighbours + the number of values received
        # The depth value is not directly sent to the all the neighbouring pixels after the tde2d
        # because thi sway is easier to count all the values received
        avg2avg_conv_weights = np.ones(conv_weights_shape, dtype=np.int32)
        #avg2avg_conv_weights[:, 1, 1, :] = 0  # TODO change to be more general
        self.val_td2avg = Conv(
            input_shape=shape_conv,
            weight=avg2avg_conv_weights,
            padding=1,
            use_graded_spike=True
        )
        self.n_td2avg = Conv(
            input_shape=shape_conv,
            weight=avg2avg_conv_weights,
            padding=1,
            use_graded_spike=True
        )

        self.td.d_out.reshape(shape_conv).connect(self.val_td2avg.s_in)
        self.td.n_out.reshape(shape_conv).connect(self.n_td2avg.s_in)

        self.val_td2avg.a_out.reshape(shape).connect(self.average_layer.s_in)
        self.n_td2avg.a_out.reshape(shape).connect(self.average_layer.n_in)
        # DEBUG
        self.average_layer.debug_out.connect(proc.out_ports.debug_out)
        self.average_layer.avg_debug.connect(proc.out_ports.avg_debug)
        # DEBUG

        # connect the outputs
        self.td.u_out.connect(proc.out_ports.u_out)
        self.td.v_out.connect(proc.out_ports.v_out)
        self.td.d_out.connect(proc.out_ports.d_out)
        self.average_layer.s_out.connect(proc.out_ports.avg_out)

        proc.vars.counter.alias(self.td.counter)
