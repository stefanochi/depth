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

class ChangeDim(AbstractProcess):
    """Time difference encoder
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        col_del = np.array(kwargs.get("col_del", []))
        row_del = np.array(kwargs.get("row_del", []))
        weight = kwargs.get("weight", 1)

        self.out_shape = (shape[0] - len(row_del), (shape[1] - len(col_del)))

        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=self.out_shape)
        self.col_del = Var(shape=col_del.shape, init=col_del)
        self.row_del = Var(shape=row_del.shape, init=row_del)
        self.weight = Var(shape=(1,), init=weight)



@implements(proc=ChangeDim, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyChangeDimModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=1)
    row_del: np.ndarray = LavaPyType(np.ndarray, int)
    col_del: np.ndarray = LavaPyType(np.ndarray, int)
    weight: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        s_in_data = self.s_in.recv()

        s_in_data = np.delete(s_in_data, self.row_del, 0)
        s_in_data = np.delete(s_in_data, self.col_del, 1)

        self.s_out.send(s_in_data * self.weight)