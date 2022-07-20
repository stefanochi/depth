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
        buffer = kwargs.get('buffer')
        self.a_in = InPort(shape=shape)

        self.events_data = Var(shape=(1, buffer), init=np.empty((1, buffer), dtype=object))


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
    events_data: np.ndarray = LavaPyType(np.ndarray, object)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        buffer = self.events_data.shape[-1]
        data_sparse = sparse.csr_matrix(a_in_data)
        self.events_data[..., self.time_step % buffer] = data_sparse


@implements(proc=EventsSink, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyEventsSinkModelFixed(PyLoihiProcessModel):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    ----------------
    a_in: signed 16-bit integer, facilitating spike. From camera, could be just 1 bit(?). Is it fixed in hw?
    t_in: signed 16-bit integer, trigger spike.
    s_out: unsigned 16-bit integer. Output time difference. What is the maximum precision?32?

    counter: unsigned 24-bit integer. The present state of the counters
    sign: the sign of the last event. Used for comparison.
    """
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    t_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)

    counter: np.ndarray = LavaPyType(np.ndarray, np.uint32, precision=24)
    sign: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=2)

    def run_spk(self):
        trig_in_data = self.t_in.recv()
        a_in_data = self.a_in.recv()

        # check if trigger is received, it has to have the same sign as the facilitating spike
        equal_sign_idx = np.equal(
            np.sign(trig_in_data),
            self.sign
        )
        # m = (np.sign(trig_in_data) == self.sign) * (self.sign != 0)
        s_out_data = self.counter * equal_sign_idx
        # reset the counter
        self.counter[equal_sign_idx] = 0.0

        # increase the counters that are started
        self.counter[self.counter > 0] += 1

        # if a new spike arrives (a_in) start the counter (or restart) and store the sign
        # If a trigger arrived in the same timestep, the counter is not started
        new_counter_idx = np.logical_and(
            np.sign(a_in_data) != 0,  # incoming facilitating spike
            np.sign(trig_in_data) == 0)  # trigger spike not arriving at the same time
        self.sign[new_counter_idx] = np.sign(a_in_data[new_counter_idx])  # get the sign of the facilitating spike
        # start the counter
        self.counter[new_counter_idx] = 1

        self.s_out.send(s_out_data)
