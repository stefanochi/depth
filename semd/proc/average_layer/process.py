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
        min_meas = kwargs.get("min_meas", 1)
        avg_alpha = kwargs.get("avg_alpha", 0.25)

        self.s_in = InPort(shape=shape)
        self.n_in = InPort(shape=shape)
        self.trig_in = InPort(shape=shape)

        self.avg_out = OutPort(shape=shape)
        # DEBUG
        self.debug_out = OutPort(shape=shape)
        self.avg_debug = OutPort(shape=shape)
        # DEBUG
        self.mean_thr = Var(shape=(1,), init=mean_thr)
        self.mean = Var(shape=shape, init=np.ones(shape) * (-1))
        self.samples = Var(shape=shape, init=np.zeros(shape))
        self.min_meas = Var(shape=(1,), init=min_meas)
        self.avg_alpha = Var(shape=(1,), init=avg_alpha)
        alpha_shift = np.log2(1.0 / avg_alpha).astype(np.int32)
        self.alpha_shift = Var(shape=(1,), init=alpha_shift)


@implements(proc=AverageLayer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAverageLayerModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    n_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    trig_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    avg_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    # DEBUG
    debug_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    avg_debug: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    # DEBUG
    mean_thr: float = LavaPyType(float, float)
    mean: np.ndarray = LavaPyType(np.ndarray, float)
    samples: np.ndarray = LavaPyType(np.ndarray, float)
    min_meas: float = LavaPyType(float, float)
    avg_alpha: float = LavaPyType(float, float)
    alpha_shift: float = LavaPyType(float, float)

    def run_spk(self):
        trig_in_data = self.trig_in.recv()
        s_in_data = self.s_in.recv()
        n_in_data = self.n_in.recv()

        # update the mean
        # with the s_in values
        m = s_in_data != 0.0
        mean_initialized = self.mean != -1.0
        # update mean where initialized
        to_update = np.logical_and(m, mean_initialized)
        self.mean[to_update] += self.avg_alpha * ((s_in_data[to_update] / n_in_data[to_update]) - self.mean[to_update])
        # initialize mean when not
        to_initialize = np.logical_and(m, np.logical_not(mean_initialized))
        self.mean[to_initialize] = (s_in_data[to_initialize] / n_in_data[to_initialize])

        self.samples[m] += n_in_data[m]

        # DEBUG
        # create list of inputs
        input_list = np.zeros_like(s_in_data)
        input_list[m] += s_in_data[m] / n_in_data[m]
        # DEBUG

        # check if the trig is close tho the mean
        m_trig = np.logical_and(np.abs(self.mean - trig_in_data) < self.mean_thr, trig_in_data != 0.0)

        # send the filtered depths
        s_out_data = self.mean * m_trig

        self.avg_out.send(s_out_data)
        # DEBUG
        self.debug_out.send(input_list)
        self.avg_debug.send(self.mean)
        # DEBUG


@implements(proc=AverageLayer, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyAverageLayerModelFixed(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    n_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    trig_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)

    avg_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    mean: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    samples: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    # DEBUG
    debug_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    avg_debug: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    # DEBUG
    mean_thr: np.int32 = LavaPyType(int, np.int32, precision=16)
    min_meas: np.int32 = LavaPyType(int, np.int32, precision=16)

    avg_alpha: np.int32 = LavaPyType(int, np.int32, precision=16)
    alpha_shift: np.int32 = LavaPyType(int, np.int32, precision=16)

    def run_spk(self):
        trig_in_data = self.trig_in.recv()
        s_in_data = self.s_in.recv()
        n_in_data = self.n_in.recv()

        # update the mean with the s_in values
        # TODO fix point division
        avg_input_idx = s_in_data != 0
        mean_initialized = self.mean != -1
        # update mean when initialized
        to_update = np.logical_and(avg_input_idx, mean_initialized)
        self.mean[to_update] += np.right_shift(
            (s_in_data[to_update] / n_in_data[to_update]).astype(np.int32) -
            self.mean[to_update], self.alpha_shift)

        # self.mean[to_update] += 1 * np.sign(
        #     (s_in_data[to_update] / n_in_data[to_update]).astype(np.int32) -
        #     self.mean[to_update])

        # initialize mean
        to_initialize = np.logical_and(avg_input_idx, np.logical_not(mean_initialized))
        self.mean[to_initialize] = (s_in_data[to_initialize] / n_in_data[to_initialize]).astype(np.int32)

        self.samples[avg_input_idx] += n_in_data[avg_input_idx]

        # DEBUG
        # create list of inputs
        input_list = np.zeros_like(s_in_data)
        input_list[avg_input_idx] += (s_in_data[avg_input_idx] / n_in_data[avg_input_idx]).astype(np.int32)
        # DEBUG

        # check if the trig is close tho the mean, there must be more than min_samples
        m_trig = np.logical_and(np.abs(self.mean - trig_in_data) < self.mean_thr, trig_in_data != 0)

        s_out_data = self.mean * (trig_in_data > 0)
        # s_out_data = trig_in_data * m_trig

        self.avg_out.send(s_out_data)
        # DEBUG
        self.debug_out.send(input_list)
        self.avg_debug.send(self.mean)
        # DEBUG
