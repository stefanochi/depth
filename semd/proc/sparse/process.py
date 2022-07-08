# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


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
