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
from lava.magma.core.decorator import implements, tag, requires
from lava.magma.core.resources import CPU


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
        conv_padding = kwargs.pop("conv_padding", (0, 0))

        conv_weights = np.zeros((1, conv_shape[0], conv_shape[1], 1))
        conv_weights[0, :, :, 0] = weights * bias_weight
        conv = Conv(
            input_shape=(input_shape[0], input_shape[1], 1),
            weight=conv_weights,
            stride=conv_stride,
            padding=conv_padding
        )

        self.out_shape = (conv.output_shape[0], conv.output_shape[1])

        self.s_in = InPort(shape=input_shape)
        self.weights = Var(shape=conv_shape, init=weights)
        self.s_out = OutPort(shape=self.out_shape)

@implements(proc=ReshapeConv, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
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
        conv_padding = proc.init_args.get("conv_padding", (0, 0))

        #self.out_shape = (int(input_shape[0]/conv_stride[0]), int(input_shape[1]/conv_stride[1]))
        conv_weights = np.zeros((1, conv_shape[0], conv_shape[1], 1))
        conv_weights[0, :, : , 0] = weights * bias_weight

        self.conv = Conv(
            input_shape=(input_shape[0], input_shape[1], 1),
            weight=conv_weights,
            stride=conv_stride,
            padding=conv_padding,
            use_graded_spike=True
        )
        self.out_shape = (self.conv.output_shape[0], self.conv.output_shape[1])

        #connect the processes together
        proc.in_ports.s_in.reshape((input_shape[0], input_shape[1], 1)).connect(self.conv.in_ports.s_in)
        self.conv.out_ports.a_out.reshape(self.out_shape).connect(proc.out_ports.s_out)
