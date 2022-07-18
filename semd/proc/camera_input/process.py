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


class CameraInputLayer(AbstractProcess):
    """Time difference encoder
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        focal_length = kwargs.get("focal_length", (1,))
        center_x = kwargs.get("center_x", (1,))
        center_y = kwargs.get("center_y", (1,))

        self.s_in = InPort(shape=(3,))
        self.x_out = OutPort(shape=shape)
        self.y_out = OutPort(shape=shape)

        self.focal_length = Var(shape=(1,), init=focal_length)

        m = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        xs = m[0] - center_x
        ys = np.flip(m[1]) - center_y

        self.px = Var(shape=shape, init=xs)
        self.py = Var(shape=shape, init=ys)


@implements(proc=CameraInputLayer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyCameraInputLayerModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    x_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    y_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    focal_length: float = LavaPyType(float, float)
    px: np.ndarray = LavaPyType(np.ndarray, float)
    py: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        s_in_data = self.s_in.recv()

        x_out_data = self.px * s_in_data[2] - self.focal_length * s_in_data[0]
        y_out_data = self.py * s_in_data[2] - self.focal_length * s_in_data[1]

        self.x_out.send(x_out_data)
        self.y_out.send(y_out_data)


@implements(proc=CameraInputLayer, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyCameraInputLayerModelFixed(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    x_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    y_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    focal_length: float = LavaPyType(float, float)
    px: np.ndarray = LavaPyType(np.ndarray, float)
    py: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        s_in_data = self.s_in.recv()

        x_out_data = self.px * s_in_data[2] - self.focal_length * s_in_data[0]
        y_out_data = self.py * s_in_data[2] - self.focal_length * s_in_data[1]

        x_out_data = x_out_data.astype(np.int32)
        y_out_data = y_out_data.astype(np.int32)

        self.x_out.send(x_out_data)
        self.y_out.send(y_out_data)
