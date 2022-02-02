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

class TDE(AbstractProcess):
    """Time difference encoder
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias = kwargs.pop("bias", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)
        trig = kwargs.pop("trig", 0)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias = Var(shape=shape, init=bias)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)

        self.t_in = InPort(shape=shape)
        self.trig = Var(shape=shape, init=trig)

@implements(proc=TDE, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyTdeModelFloat(PyLoihiProcessModel):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    """
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)
    vth: float = LavaPyType(float, float)

    t_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    trig: bool = LavaPyType(bool, bool)

    def run_spk(self):
        t_in_data = self.t_in.recv()
        #self.trig = np.logical_or(t_in_data, self.trig)

        a_in_data = self.a_in.recv() * 10
        self.u[:] = self.u * (1 - self.du)
        a = a_in_data > np.zeros(t_in_data.shape)
        self.u[a] = a_in_data[a]
        #m = a_in_data
        # bias = self.bias * (2**self.bias_exp)
        # self.v[:] = self.v * (1 - self.dv) + self.u + bias
        # s_out = np.logical_and(self.v >= self.vth, self.trig)
        # self.v[s_out] = 0  # Reset voltage to 0

        self.s_out.send(t_in_data * self.u)
        f = t_in_data > np.zeros(t_in_data.shape)
        self.u[f] = 0