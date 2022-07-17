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
        self.vth = Var(shape=(1,), init=vth)
        # DEBUG
        self.debug_out = OutPort(shape=self.out_shape)
        self.avg_debug = OutPort(shape=self.out_shape)
        # DEBUG
        self.initialize_weights()
        self.initialize_conv_weights(avg_conv_shape)

    def initialize_weights(self):
        up_weights = utils.get_connection_up(self.out_shape)
        down_weights = utils.get_connection_down(self.out_shape)
        left_weights = utils.get_connection_left(self.out_shape)
        right_weights = utils.get_connection_right(self.out_shape)

        self.up_weights = Var(shape=self.out_shape, init=up_weights)
        self.down_weights = Var(shape=self.out_shape, init=down_weights)
        self.left_weights = Var(shape=self.out_shape, init=left_weights)
        self.right_weights = Var(shape=self.out_shape, init=right_weights)
        n_elems = self.out_shape[0] * self.out_shape[1]
        conn_shape = (n_elems, n_elems)
        self.trig_weights = Var(shape=(conn_shape), init=np.eye(n_elems, n_elems))

    def initialize_conv_weights(self, avg_conv_shape):
        n_elems = self.out_shape[0] * self.out_shape[1]
        conv = Convolution(np.full(avg_conv_shape, 1.0))
        conv.configure(self.out_shape)
        conv_weights = conv._compute_weights() - np.eye(n_elems)
        self.conv_avg_weights = Var(shape=conv_weights.shape, init=conv_weights)


@implements(proc=Semd2dLayer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
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

        up_weights = proc.vars.up_weights.init
        down_weights = proc.vars.down_weights.init
        left_weights = proc.vars.left_weights.init
        right_weights = proc.vars.right_weights.init
        trig_weights = proc.vars.trig_weights.init
        conv_avg_weights = proc.vars.conv_avg_weights.init

        # convolution layer with reshape proc
        # this layer performs the subsampling and is the connection layer
        # between the input and the network
        # TODO option to remove subsampling layer
        bias_weight = (1.0 / (conv_shape[0] * conv_shape[1])) / thresh_conv
        self.conv = ReshapeConv(input_shape=shape, conv_shape=conv_shape, conv_stride=conv_stride,
                                bias_weight=bias_weight)

        out_shape = self.conv.out_shape
        # connect the input to the convolution
        proc.in_ports.s_in.connect(self.conv.in_ports.s_in)

        # lif layer that collects the value from the convolution layer and fires when enough neuron in the input fire
        self.lif = TernaryLIF(shape=out_shape, du=1.0, dv=0.0, vth_hi=vth, vth_lo=-vth, use_graded_spike=True)

        # conv to lif
        self.conv.s_out.connect(self.lif.a_in)

        # TDE layer, it contains the TDE for all the 4 directions
        self.td = TDE2D(shape=out_shape)

        proc.in_ports.tu_in.connect(self.td.t_u)
        proc.in_ports.tv_in.connect(self.td.t_v)

        n_elems = out_shape[0] * out_shape[1]
        conn_shape = (n_elems, n_elems)

        # create all the connections for the TDE2D
        self.dense_up = Dense(shape=conn_shape, weights=up_weights, use_graded_spike=True)
        self.dense_down = Dense(shape=conn_shape, weights=down_weights, use_graded_spike=True)
        self.dense_left = Dense(shape=conn_shape, weights=left_weights, use_graded_spike=True)
        self.dense_right = Dense(shape=conn_shape, weights=right_weights, use_graded_spike=True)
        self.dense_trig = Dense(shape=conn_shape, weights=trig_weights, use_graded_spike=True)
        # connect to the dense layers
        self.lif.s_out.flatten().connect(self.dense_up.s_in)
        self.lif.s_out.flatten().connect(self.dense_down.s_in)
        self.lif.s_out.flatten().connect(self.dense_left.s_in)
        self.lif.s_out.flatten().connect(self.dense_right.s_in)
        self.lif.s_out.flatten().connect(self.dense_trig.s_in)
        # connect the dense to the TDE2D
        self.dense_up.a_out.reshape(out_shape).connect(self.td.up_in)
        self.dense_down.a_out.reshape(out_shape).connect(self.td.down_in)
        self.dense_left.a_out.reshape(out_shape).connect(self.td.left_in)
        self.dense_right.a_out.reshape(out_shape).connect(self.td.right_in)
        self.dense_trig.a_out.reshape(out_shape).connect(self.td.trig_in)

        # Average Layer
        self.average_layer = AverageLayer(shape=out_shape, mean_thr=avg_thresh, min_meas=avg_min_meas, avg_alpha=avg_alpha)
        # dense connections, the connection here is simply a 1two1
        self.dense_avg = Dense(shape=conn_shape, weights=trig_weights, use_graded_spike=True)
        self.td.d_out.flatten().connect(self.dense_avg.s_in)
        self.dense_avg.a_out.reshape(out_shape).connect(self.average_layer.trig_in)
        # recurrent connection in the average layer. Implemented as a convolution
        self.conv_dense = Dense(shape=conn_shape, weights=conv_avg_weights, use_graded_spike=True)
        self.n_conv_dense = Dense(shape=conn_shape, weights=conv_avg_weights, use_graded_spike=True)

        self.average_layer.avg_out.flatten().connect(self.conv_dense.s_in)
        self.conv_dense.a_out.reshape(out_shape).connect(self.average_layer.s_in)
        self.average_layer.n_out.flatten().connect(self.n_conv_dense.s_in)
        self.n_conv_dense.a_out.reshape(out_shape).connect(self.average_layer.n_in)
        # DEBUG
        self.average_layer.debug_out.connect(proc.out_ports.debug_out)
        self.average_layer.avg_debug.connect(proc.out_ports.avg_debug)
        # DEBUG

        # connect the outputs
        self.td.u_out.connect(proc.out_ports.u_out)
        self.td.v_out.connect(proc.out_ports.v_out)
        self.td.d_out.connect(proc.out_ports.d_out)
        self.average_layer.s_out.connect(proc.out_ports.avg_out)
