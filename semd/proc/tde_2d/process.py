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
from lava.magma.core.model.sub.model import AbstractSubProcessModel

from semd.proc.tde.process import TDE
from semd.proc.extract_flow.process import ExtractFlow

class TDE2D(AbstractProcess):
    """Time difference encoder for the 2D version.
    It receives 4 inputs, one for each xy direction.
    Plus one more input when the center pixels receives an event.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))

        self.shape = shape

        self.up_in = InPort(shape=shape)
        self.down_in = InPort(shape=shape)
        self.left_in = InPort(shape=shape)
        self.right_in = InPort(shape=shape)

        self.trig_in = InPort(shape=shape)

        self.u_out = OutPort(shape=shape)
        self.v_out = OutPort(shape=shape)
        self.d_out = OutPort(shape=shape)
        self.n_out = OutPort(shape=shape)

        self.t_u = InPort(shape=shape)
        self.t_v = InPort(shape=shape)

        self.counter = Var(shape=shape, init=0)


@implements(proc=TDE2D, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt', 'fixed_pt')
class PyTde2dModelFloat(AbstractSubProcessModel):
    """
    """
    def __init__(self, proc):
        shape = proc.init_args.get("shape", (1,))

        self.up_tde = TDE(shape=shape)
        self.down_tde = TDE(shape=shape)
        self.right_tde = TDE(shape=shape)
        self.left_tde = TDE(shape=shape)

        proc.in_ports.up_in.connect(self.up_tde.a_in)
        proc.in_ports.down_in.connect(self.down_tde.a_in)
        proc.in_ports.right_in.connect(self.right_tde.a_in)
        proc.in_ports.left_in.connect(self.left_tde.a_in)

        proc.in_ports.trig_in.connect(self.up_tde.t_in)
        proc.in_ports.trig_in.connect(self.down_tde.t_in)
        proc.in_ports.trig_in.connect(self.right_tde.t_in)
        proc.in_ports.trig_in.connect(self.left_tde.t_in)

        self.flow = ExtractFlow(shape=shape)

        proc.in_ports.t_u.connect(self.flow.t_u)
        proc.in_ports.t_v.connect(self.flow.t_v)

        self.up_tde.s_out.connect(self.flow.up_in)
        self.down_tde.s_out.connect(self.flow.down_in)
        self.right_tde.s_out.connect(self.flow.right_in)
        self.left_tde.s_out.connect(self.flow.left_in)

        self.flow.u_out.connect(proc.out_ports.u_out)
        self.flow.v_out.connect(proc.out_ports.v_out)
        self.flow.d_out.connect(proc.out_ports.d_out)
        self.flow.n_out.connect(proc.out_ports.n_out)

        proc.vars.counter.alias(self.up_tde.counter)
