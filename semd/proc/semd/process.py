import numpy as np

from lava.proc.lif.process import LIF
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.proc.io.source import RingBuffer

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements

from semd.proc.tde.process import TDE
from semd.proc.reshape_conv.process import ReshapeConv
from semd.proc.change_dim.process import ChangeDim


class SemdLayer(AbstractProcess):
    """sEMD
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape", (1, 1))
        conv_shape = kwargs.pop("conv_shape", (1, 1))
        conv_stride = kwargs.pop("conv_stride", (1, 1))
        thresh_conv = kwargs.pop("thresh_conv", 0.5)
        vth = kwargs.pop("vth", 10)
        detector_du = kwargs.pop("detector_du", 0.05)

        bias_weight = vth / conv_shape[0] * conv_shape[1] * thresh_conv
        # convolution layer with reshape proc
        conv = ReshapeConv(input_shape=shape, conv_shape=conv_shape, conv_stride=conv_stride,
                                bias_weight=bias_weight)

        self.out_shape = conv.out_shape
        self.detector_shape = (self.out_shape[0], self.out_shape[1] - 1)

        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=self.detector_shape)
        self.vth = Var(shape=(1,), init=vth)


@implements(proc=SemdLayer, protocol=LoihiProtocol)
class SemdLayerModel(AbstractSubProcessModel):

    def __init__(self, proc):
        # input layer
        shape = proc.init_args.get("shape", (1, 1))
        conv_shape = proc.init_args.get("conv_shape", (1, 1))
        conv_stride = proc.init_args.get("conv_stride", (1, 1))
        thresh_conv = proc.init_args.get("thresh_conv", 0.5)
        vth = proc.init_args.get("vth", 10)
        detector_du = proc.init_args.get("detector_du", 0.05)

        bias_weight = (vth / (conv_shape[0] * conv_shape[1])) / thresh_conv
        # convolution layer with reshape proc
        self.conv = ReshapeConv(input_shape=shape, conv_shape=conv_shape, conv_stride=conv_stride,
                                bias_weight=bias_weight)

        out_shape = self.conv.out_shape
        detector_shape = (out_shape[0], out_shape[1] - 1)

        # lif layer that collects the value from the convolution layer and fires when enough neuron in the input fire
        self.lif = LIF(shape=out_shape, du=1.0, dv=0.0, vth=10.0)

        # removes columns and row from the lif to match the dimension of the time difference detector
        self.change_dim_excit = ChangeDim(shape=(out_shape), col_del=[1], row_del=[])
        self.change_dim_trig = ChangeDim(shape=(out_shape), col_del=[-1], row_del=[])

        # time difference detector for the left motion
        # TODO implement other directions
        self.left_detector = TDE(shape=detector_shape, du=detector_du)

        # connect the input to the convolution
        proc.in_ports.s_in.connect(self.conv.in_ports.s_in)
        # conv to lif
        self.conv.s_out.connect(self.lif.a_in)

        self.lif.s_out.connect(self.change_dim_excit.s_in)
        self.lif.s_out.connect(self.change_dim_trig.s_in)

        self.change_dim_trig.s_out.connect(self.left_detector.t_in)
        self.change_dim_excit.s_out.connect(self.left_detector.a_in)

        self.left_detector.s_out.connect(proc.out_ports.s_out)
