# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import scipy.sparse as sparse

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.dense.process import Dense


@implements(proc=Dense, protocol=LoihiProtocol)
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