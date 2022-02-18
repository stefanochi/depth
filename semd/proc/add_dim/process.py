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

class AddDim(AbstractProcess):
    """Time difference encoder
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        out_shape = kwargs.get("out_shape", (1,))
        in_shape = kwargs.get("in_shape", (1,))
        dis_x = kwargs.get("dis_x", 0)
        dis_y = kwargs.get("dis_y", 0)

        self.s_in = InPort(shape=in_shape)
        self.s_out = OutPort(shape=out_shape)
        self.dis_x = Var(shape=(1,), init=dis_x)
        self.dis_y = Var(shape=(1,), init=dis_y)


@implements(proc=AddDim, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAddDimModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=1)
    dis_x: int = LavaPyType(int, int)
    dis_y: int = LavaPyType(int, int)

    def run_spk(self):
        s_in_data = self.s_in.recv()

        out_shape = self.s_out._shape
        out_data = np.zeros(out_shape)

        if self.dis_x >= 0:
            if self.dis_y >= 0:
                out_data[self.dis_y:,self.dis_x:] = s_in_data
            else:
                out_data[:self.dis_y, self.dis_x:] = s_in_data
        else:
            if self.dis_y >= 0:
                out_data[self.dis_y:,:self.dis_x] = s_in_data
            else:
                out_data[:self.dis_y,:self.dis_x] = s_in_data

        self.s_out.send(out_data)