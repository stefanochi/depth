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

class SelExp(AbstractProcess):
    """Time difference encoder
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape_in = kwargs.get("shape_in", (1,))
        out_r = kwargs.get("out_r", (1,))
        out_c = kwargs.get("out_c", (1,))
        sel_id = kwargs.get("sel_id", (1,))
        weight = kwargs.get("weight", 1)

        self.s_in = InPort(shape=shape_in)
        self.s_out = OutPort(shape=(out_r, out_c))
        self.weight = Var(shape=(1,), init=weight)
        self.sel_id = Var(shape=(1,), init=sel_id)
        self.out_r = Var(shape=(1,), init=out_r)
        self.out_c = Var(shape=(1,), init=out_c)



@implements(proc=SelExp, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySelExpModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=1)
    sel_id: int = LavaPyType(int, int)
    weight: np.ndarray = LavaPyType(np.ndarray, float)
    out_r: int = LavaPyType(int, int)
    out_c: int = LavaPyType(int, int)

    def run_spk(self):
        s_in_data = self.s_in.recv()

        data_out = np.ones((self.out_r, self.out_c)) * s_in_data[self.sel_id]

        self.s_out.send(data_out * self.weight)