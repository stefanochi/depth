from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

import numpy as np
import scipy.sparse as sparse
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel


class EventsSink(AbstractProcess):
    """record output events
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.a_in = InPort(shape=shape)

        self.events_data = Var(shape=shape, init=np.empty(shape))


@implements(proc=EventsSink, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyEventsSinkModelFloat(PyLoihiProcessModel):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    """
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    events_data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.events_data = a_in_data


@implements(proc=EventsSink, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyEventsSinkModelFixed(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    events_data: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.events_data = a_in_data
