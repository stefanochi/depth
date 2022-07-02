from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel


class ExtractFlow(AbstractProcess):
    """Time difference encoder for the 2D version.
    It receives 4 inputs, one for each xy direction.
    Plus one more input when the center pixels receives an event.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        tu = kwargs.get("tu", (1,))
        tv = kwargs.get("tv", (1,))

        self.shape = shape
        # variables that hold the counter for
        # horizontal and vertical motion

        self.up_in = InPort(shape=shape)
        self.down_in = InPort(shape=shape)
        self.left_in = InPort(shape=shape)
        self.right_in = InPort(shape=shape)

        self.u_out = OutPort(shape=shape)
        self.v_out = OutPort(shape=shape)
        self.d_out = OutPort(shape=shape)

        self.t_u = Var(shape=(1, ), init=tu)
        self.t_v = Var(shape=(1, ), init=tv)


@implements(proc=ExtractFlow, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyExtractFlowModelFloat(PyLoihiProcessModel):
    """
    """
    up_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    down_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    left_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    right_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    u_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    v_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    d_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    t_u: float = LavaPyType(float, float)
    t_v: float = LavaPyType(float, float)

    def run_spk(self):
        # input from each direction
        up_td_data = self.up_in.recv()
        down_td_data = self.down_in.recv()
        left_td_data = self.left_in.recv()
        right_td_data = self.right_in.recv()

        v_td_out = np.zeros(up_td_data.shape)
        u_m = np.logical_and(
            up_td_data > 0,
            np.logical_or(up_td_data < down_td_data, np.isclose(down_td_data, 0.0))
        )
        v_td_out[u_m] = up_td_data[u_m]
        d_m = np.logical_and(
            down_td_data > 0,
            np.logical_or(down_td_data < up_td_data, np.isclose(up_td_data, 0.0))
        )
        v_td_out[d_m] = -down_td_data[d_m]

        h_td_out = np.zeros(right_td_data.shape)
        r_m = np.logical_and(
            right_td_data > 0,
            np.logical_or(right_td_data < left_td_data, np.isclose(left_td_data, 0.0))
        )
        h_td_out[r_m] = right_td_data[r_m]
        l_m = np.logical_and(
            left_td_data > 0,
            np.logical_or(left_td_data < right_td_data, np.isclose(right_td_data, 0.0))
        )
        h_td_out[l_m] = -left_td_data[l_m]

        self.u_out.send(h_td_out)
        self.v_out.send(v_td_out)

        d = h_td_out * self.t_u + v_td_out * self.t_v
        d *= d > 0.0
        self.d_out.send(d)
