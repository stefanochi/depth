import numpy as np
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.proc.conv.process import Conv
from lava.proc.conv import utils

from lava.lib.dnf.connect.reshape_int.process import ReshapeInt

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements


class ReshapeConv(AbstractProcess):
    """sEMD
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        input_shape = kwargs.pop("input_shape", (1, 1))
        conv_shape = kwargs.pop("conv_shape", (3, 3))
        weights = kwargs.pop("weights", np.ones((conv_shape)))
        conv_stride = kwargs.pop("conv_stride", (3, 3))
        bias_weight = kwargs.pop("bias_weight", 1)

        conv_weights = np.zeros((1, conv_shape[0], conv_shape[1], 1))
        conv_weights[0, :, :, 0] = weights * bias_weight
        conv = Conv(
            input_shape=(input_shape[0], input_shape[1], 1),
            weight=conv_weights,
            stride=conv_stride
        )

        self.out_shape = (conv.output_shape[0], conv.output_shape[1])

        self.s_in = InPort(shape=input_shape)
        self.weights = Var(shape=conv_shape, init=weights)
        self.s_out = OutPort(shape=self.out_shape)

@implements(proc=ReshapeConv, protocol=LoihiProtocol)
class ReshapeConvModel(AbstractSubProcessModel):

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        # Instantiate child processes
        # input shape is a 2D vec (shape of weight mat)
        input_shape = proc.init_args.get("input_shape", (1, 1))
        conv_shape = proc.init_args.get("conv_shape", (3, 3))
        weights = proc.init_args.get("weights", np.ones((conv_shape)))
        conv_stride = proc.init_args.get("conv_stride", (3, 3))
        bias_weight = proc.init_args.get("bias_weight", 1)

        #self.out_shape = (int(input_shape[0]/conv_stride[0]), int(input_shape[1]/conv_stride[1]))
        conv_weights = np.zeros((1, conv_shape[0], conv_shape[1], 1))
        conv_weights[0, :, : , 0] = weights * bias_weight

        self.conv = Conv(
            input_shape=(input_shape[0], input_shape[1], 1),
            weight=conv_weights,
            stride=conv_stride
        )
        self.out_shape = (self.conv.output_shape[0], self.conv.output_shape[1])
        self.input_reshape = ReshapeInt(shape_in=input_shape, shape_out=(input_shape[0], input_shape[1], 1))

        self.output_reshape = ReshapeInt(shape_in=(self.out_shape[0], self.out_shape[1], 1), shape_out=self.out_shape)

        #connect the processes together
        proc.in_ports.s_in.connect(self.input_reshape.in_ports.s_in)
        self.input_reshape.out_ports.s_out.connect(self.conv.in_ports.s_in)
        self.conv.out_ports.a_out.connect(self.output_reshape.in_ports.s_in)
        self.output_reshape.out_ports.s_out.connect(proc.out_ports.s_out)