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

        self.shape = shape

        self.up_in = InPort(shape=shape)
        self.down_in = InPort(shape=shape)
        self.left_in = InPort(shape=shape)
        self.right_in = InPort(shape=shape)

        self.u_out = OutPort(shape=shape)
        self.v_out = OutPort(shape=shape)
        self.d_out = OutPort(shape=shape)

        self.t_u = InPort(shape=shape)
        self.t_v = InPort(shape=shape)


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

    t_u: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    t_v: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def run_spk(self):
        # input from each direction
        up_td_data = self.up_in.recv()
        down_td_data = self.down_in.recv()
        left_td_data = self.left_in.recv()
        right_td_data = self.right_in.recv()

        t_u = self.t_u.recv()
        t_v = self.t_v.recv()

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

        # TODO change to store the values so they don't need to be sent at every timestep
        d = h_td_out * t_u + v_td_out * t_v
        d *= d > 0.0
        self.d_out.send(d)


@implements(proc=ExtractFlow, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyExtractFlowModelFixed(PyLoihiProcessModel):
    """
    """
    up_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    down_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    left_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    right_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)

    u_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    v_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    d_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)

    t_u: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    t_v: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)

    def run_spk(self):
        # input from each direction
        up_td_data = self.up_in.recv()
        down_td_data = self.down_in.recv()
        left_td_data = self.left_in.recv()
        right_td_data = self.right_in.recv()

        t_u = self.t_u.recv()
        t_v = self.t_v.recv()

        v_td_out = np.zeros(up_td_data.shape, dtype=np.int32)
        u_m = np.logical_and(
            up_td_data > 0,
            np.logical_or(up_td_data < down_td_data, down_td_data == 0)
        )
        v_td_out[u_m] = up_td_data[u_m]
        d_m = np.logical_and(
            down_td_data > 0,
            np.logical_or(down_td_data < up_td_data, up_td_data == 0)
        )
        v_td_out[d_m] = -down_td_data[d_m]

        h_td_out = np.zeros(right_td_data.shape, dtype=np.int32)
        r_m = np.logical_and(
            right_td_data > 0,
            np.logical_or(right_td_data < left_td_data, up_td_data == 0)
        )
        h_td_out[r_m] = right_td_data[r_m]
        l_m = np.logical_and(
            left_td_data > 0,
            np.logical_or(left_td_data < right_td_data, right_td_data == 0)
        )
        h_td_out[l_m] = -left_td_data[l_m]

        self.u_out.send(h_td_out)
        self.v_out.send(v_td_out)

        # TODO change to store the values so they don't need to be sent at every timestep
        d = h_td_out * t_u + v_td_out * t_v
        d *= d > 0  # ignore negatives
        self.d_out.send(d)
