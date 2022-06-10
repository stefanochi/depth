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


class TDE2D(AbstractProcess):
    """Time difference encoder for the 2D version.
    It receives 4 inputs, one for each xy direction.
    Plus one more input when the center pixels receives an event.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))

        self.shape = shape
        # variables that hold the counter for
        # horizontal and vertical motion
        self.h_td = Var(shape=shape, init=np.zeros(shape))
        self.v_td = Var(shape=shape, init=np.zeros(shape))
        # record the sign of the events arriving
        self.h_sign = Var(shape=shape, init=np.zeros(shape))
        self.v_sign = Var(shape=shape, init=np.zeros(shape))

        self.up_in = InPort(shape=shape)
        self.down_in = InPort(shape=shape)
        self.left_in = InPort(shape=shape)
        self.right_in = InPort(shape=shape)

        self.trig_in = InPort(shape=shape)

        self.u_out = OutPort(shape=shape)
        self.v_out = OutPort(shape=shape)
        self.d_out = OutPort(shape=shape)


@implements(proc=TDE2D, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyTde2dModelFloat(PyLoihiProcessModel):
    """
    """
    up_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    down_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    left_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    right_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    h_td: np.ndarray = LavaPyType(np.ndarray, float)
    v_td: np.ndarray = LavaPyType(np.ndarray, float)
    h_sign: np.ndarray = LavaPyType(np.ndarray, float)
    v_sign: np.ndarray = LavaPyType(np.ndarray, float)

    trig_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    u_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    v_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    d_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        # input from each direction
        up_in_data = self.up_in.recv()
        down_in_data = self.down_in.recv()
        left_in_data = self.left_in.recv()
        right_in_data = self.right_in.recv()

        # input from the trigger pixel
        trig_in_data = self.trig_in.recv()

        # if the trigger arrives, reset the counter and output the time
        # difference measured.
        m = (trig_in_data == self.h_sign) * (self.h_sign != 0.0)
        h_td_out = self.h_td * m
        self.h_td[m] = 0.0
        m = (trig_in_data == self.v_sign) * (self.v_sign != 0.0)
        v_td_out = self.v_td * m
        self.v_td[m] = 0.0

        # increase each counter if it is more than zero
        # if td is zeros, no input was received and the counter
        # does not need to be increased
        self.v_td[self.v_td > 0] += 1
        self.v_td[self.v_td < 0] -= 1

        self.h_td[self.h_td > 0] += 1
        self.h_td[self.h_td < 0] -= 1

        # TODO what happens when both input directions fire a the same time?
        # store the polarity of the event that starts the counter
        self.v_sign = self.v_sign * (up_in_data == 0) + (up_in_data * (trig_in_data == 0.0))
        self.v_sign = self.v_sign * (down_in_data == 0) + (down_in_data * (trig_in_data == 0.0))
        self.h_sign = self.h_sign * (right_in_data == 0) + (right_in_data * (trig_in_data == 0.0))
        self.h_sign = self.h_sign * (left_in_data == 0) + (left_in_data * (trig_in_data == 0.0))
        # start the counter if one of the input arrives,
        # but ignore the input if the corresponding counter is already active
        # If a trigger arrived in the same timestep, the counter is not started
        self.v_td = self.v_td * (up_in_data == 0.0) + (np.abs(up_in_data) * (trig_in_data == 0.0))
        self.v_td = self.v_td * (down_in_data == 0.0) - (np.abs(down_in_data) * (trig_in_data == 0.0))
        self.h_td = self.h_td * (right_in_data == 0.0) + (np.abs(right_in_data) * (trig_in_data == 0.0))
        self.h_td = self.h_td * (left_in_data == 0.0) - (np.abs(left_in_data) * (trig_in_data == 0.0))

        # send the output spikes
        # for now as two difference variable TODO change

        self.u_out.send(h_td_out)
        self.v_out.send(v_td_out)

        d = h_td_out * 0.002 * (-103) + v_td_out * 0.002 * (0)
        d *= d > 0.0
        self.d_out.send(d)
