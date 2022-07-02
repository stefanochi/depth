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


class AverageLayer(AbstractProcess):
    """Time difference encoder
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        mean_thr = kwargs.get("mean_thr", 0.1)

        self.s_in = InPort(shape=shape)
        self.n_in = InPort(shape=shape)
        self.trig_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.n_out = OutPort(shape=shape)
        self.mean_thr = Var(shape=(1,), init=mean_thr)
        self.mean = Var(shape=shape, init=np.zeros(shape))
        self.samples = Var(shape=shape, init=np.zeros(shape))


@implements(proc=AverageLayer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAverageLayerModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    n_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    trig_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    n_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    mean_thr: float = LavaPyType(float, float)
    mean: np.ndarray = LavaPyType(np.ndarray, float)
    samples: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        s_in_data = self.s_in.recv()
        trig_in_data = self.trig_in.recv()
        n_in_data = self.n_in.recv()
        # TODO more spikes arriving at the same time
        # filter out the outliers
        m = s_in_data != 0.0
        self.mean[m] = ((self.mean[m] * self.samples[m]) + s_in_data[m]) / (self.samples[m] + n_in_data[m])
        self.samples[m] += n_in_data[m]

        # check if the trig is close tho the mean
        m_trig = np.logical_or(
            np.logical_and(np.abs(self.mean - trig_in_data) < self.mean_thr, trig_in_data != 0.0),
            np.logical_and(self.samples < 5, trig_in_data != 0.0))

        # if it is close, update the mean t´with the average
        self.mean[m_trig] = ((self.mean[m_trig] * self.samples[m_trig]) + trig_in_data[m_trig]) / (
                self.samples[m_trig] + 1)
        self.samples[m_trig] += 1

        s_out_data = trig_in_data * m_trig
        n_out_data = np.full(self.mean.shape, 1.0) * m_trig
        self.s_out.send(s_out_data)
        self.n_out.send(n_out_data)
