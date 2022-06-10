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

        bias_weight = vth / conv_shape[0] * conv_shape[1] * thresh_conv
        # convolution layer with reshape proc
        conv = ReshapeConv(input_shape=shape, conv_shape=conv_shape, conv_stride=conv_stride,
                           bias_weight=bias_weight)

        self.out_shape = conv.out_shape
        self.detector_shape = (self.out_shape[0] - 2, self.out_shape[1] - 2)

        self.s_in = InPort(shape=shape)
        self.u_out = OutPort(shape=self.out_shape)
        self.v_out = OutPort(shape=self.out_shape)
        self.d_out = OutPort(shape=self.out_shape)
        self.vth = Var(shape=(1,), init=vth)


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

        bias_weight = (1.0 / (conv_shape[0] * conv_shape[1])) / thresh_conv
        # convolution layer with reshape proc
        # this layer performs the subsampling and is the connection layer
        # between the input and the network
        self.conv = ReshapeConv(input_shape=shape, conv_shape=conv_shape, conv_stride=conv_stride,
                                bias_weight=bias_weight)

        out_shape = self.conv.out_shape
        # the first and last column and the first and last row are removed.
        detector_shape = (out_shape[0] - 2, out_shape[1] - 2)
        # connect the input to the convolution
        proc.in_ports.s_in.connect(self.conv.in_ports.s_in)

        # lif layer that collects the value from the convolution layer and fires when enough neuron in the input fire
        self.lif = TernaryLIF(shape=out_shape, du=1.0, dv=0.0, vth_hi=vth, vth_lo=-vth, use_graded_spike=True)

        # conv to lif
        self.conv.s_out.connect(self.lif.a_in)

        self.td = TDE2D(shape=out_shape)

        # removes columns and row from the lif to match the dimension of the time difference detector
        # TODO change
        n_elems = out_shape[0] * out_shape[1]
        conn_shape = (n_elems, n_elems)

        self.dense_up = Dense(shape=conn_shape, weights=utils.get_connection_up(out_shape), use_graded_spike=True)
        self.dense_down = Dense(shape=conn_shape, weights=utils.get_connection_down(out_shape), use_graded_spike=True)
        self.dense_left = Dense(shape=conn_shape, weights=utils.get_connection_left(out_shape), use_graded_spike=True)
        self.dense_right = Dense(shape=conn_shape, weights=utils.get_connection_right(out_shape), use_graded_spike=True)

        # self.change_dim_excit_up = ChangeDim(shape=(out_shape), col_del=[-1, 0], row_del=[0, 1])
        # self.change_dim_excit_down = ChangeDim(shape=(out_shape), col_del=[-1, 0], row_del=[-1, -2])
        # self.change_dim_excit_left = ChangeDim(shape=(out_shape), col_del=[0, 1], row_del=[-1, 0])
        # self.change_dim_excit_right = ChangeDim(shape=(out_shape), col_del=[-1, -2], row_del=[-1, 0])

        # self.change_dim_trig = ChangeDim(shape=out_shape, col_del=[0, -1], row_del=[0, -1])
        self.dense_trig = Dense(shape=conn_shape, weights=np.eye(n_elems, n_elems), use_graded_spike=True)

        self.lif.s_out.flatten().connect(self.dense_up.s_in)
        self.lif.s_out.flatten().connect(self.dense_down.s_in)
        self.lif.s_out.flatten().connect(self.dense_left.s_in)
        self.lif.s_out.flatten().connect(self.dense_right.s_in)
        # connect the trigger
        self.lif.s_out.flatten().connect(self.dense_trig.s_in)

        self.dense_up.a_out.reshape(out_shape).connect(self.td.up_in)
        self.dense_down.a_out.reshape(out_shape).connect(self.td.down_in)
        self.dense_left.a_out.reshape(out_shape).connect(self.td.left_in)
        self.dense_right.a_out.reshape(out_shape).connect(self.td.right_in)

        self.dense_trig.a_out.reshape(out_shape).connect(self.td.trig_in)

        self.average_layer = AverageLayer(shape=out_shape, mean_thr=0.5)

        self.dense_avg = Dense(shape=conn_shape, weights=np.eye(conn_shape[0]), use_graded_spike=True)

        self.td.d_out.flatten().connect(self.dense_avg.s_in)
        self.dense_avg.a_out.reshape(out_shape).connect(self.average_layer.trig_in)

        # use convolution to connect the average layer
        # connect(self.average_layer.s_out, self.average_layer.s_in, ops=[Convolution(np.full((5, 5), 1.0))])
        conv = Convolution(np.full((5, 5), 1.0))
        conv.configure(out_shape)
        conv_weights = conv._compute_weights() - np.eye(n_elems)
        self.conv_dense = Dense(shape=conn_shape, weights=conv_weights, use_graded_spike=True)
        self.n_conv_dense = Dense(shape=conn_shape, weights=conv_weights, use_graded_spike=True)

        self.average_layer.s_out.flatten().connect(self.conv_dense.s_in)
        self.conv_dense.a_out.reshape(out_shape).connect(self.average_layer.s_in)

        self.average_layer.n_out.flatten().connect(self.n_conv_dense.s_in)
        self.n_conv_dense.a_out.reshape(out_shape).connect(self.average_layer.n_in)

        # avg_conv_shape = (5, 5)
        # avg_conv_stride = (1, 1)
        # avg_conv_weights = 1.0
        # self.avg_conv = ReshapeConv(input_shape=detector_shape, conv_shape=avg_conv_shape, conv_stride=avg_conv_stride,
        #                             bias_weight=avg_conv_weights, conv_padding=(2, 2))
        #
        # self.average_layer.s_out.connect(self.avg_conv.s_in)
        # self.avg_conv.s_out.connect(self.average_layer.s_in)

        self.td.u_out.connect(proc.out_ports.u_out)
        self.td.v_out.connect(proc.out_ports.v_out)
        self.lif.s_out.connect(proc.out_ports.d_out)

        # proc.vars.u.alias(self.detector.vars.u)
