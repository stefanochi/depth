# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import scipy.sparse as sparse

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.dense.process import Dense


class Sparse(AbstractProcess):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        if len(shape) != 2:
            raise AssertionError("Dense Process 'shape' expected a 2D tensor.")
        weights = kwargs.pop("weights", np.zeros(shape=shape))
        if len(np.shape(weights)) != 2:
            raise AssertionError("Dense Process 'weights' expected a 2D "
                                 "matrix.")
        weight_exp = kwargs.pop("weight_exp", 0)
        num_weight_bits = kwargs.pop("num_weight_bits", 8)
        sign_mode = kwargs.pop("sign_mode", 1)
        use_graded_spike = kwargs.get('use_graded_spike', False)

        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=weights)
        self.weight_exp = Var(shape=(1,), init=weight_exp)
        self.num_weight_bits = Var(shape=(1,), init=num_weight_bits)
        self.sign_mode = Var(shape=(1,), init=sign_mode)
        self.a_buff = Var(shape=(shape[0],), init=0)
        self.use_graded_spike = Var(shape=(1,), init=use_graded_spike)


@implements(proc=Sparse, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySparseModelFloat(PyLoihiProcessModel):
    """Implementation of Conn Process with Dense synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation.
    """
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons)in C-order (row major).
    weights: sparse.dia_array = LavaPyType(sparse.dia_array, float)
    weight_exp: float = LavaPyType(float, float)
    num_weight_bits: float = LavaPyType(float, float)
    sign_mode: float = LavaPyType(float, float)
    use_graded_spike: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        # The a_out sent on a each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff)
        if self.use_graded_spike.item():
            s_in = self.s_in.recv()
            s_in = sparse.csr_array(s_in)
            self.a_buff = self.weights.tocsr().dot(s_in).toarray()
        else:
            s_in = self.s_in.recv().astype(bool)
            self.a_buff = self.weights[:, s_in].sum(axis=1)
