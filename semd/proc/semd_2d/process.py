import numpy as np

from lava.proc.lif.process import LIF
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.proc.io.source import RingBuffer

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements

#from utils import calc_patch_size
from semd.proc.tde_2d.process import TDE2D
from semd.proc.reshape_conv.process import ReshapeConv
from semd.proc.change_dim.process import ChangeDim
from semd.proc.add_dim.process import AddDim


class Semd2dLayer(AbstractProcess):
    """sEMD
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape", (1, 1))
        conv_shape = kwargs.pop("conv_shape", (1, 1))
        conv_stride = kwargs.pop("conv_stride", (1, 1))
        thresh_conv = kwargs.pop("thresh_conv", 0.5)
        vth = kwargs.pop("vth", 10)
        detector_dv = kwargs.pop("detector_dv", 0.05)

        bias_weight = vth / conv_shape[0] * conv_shape[1] * thresh_conv
        # convolution layer with reshape proc
        conv = ReshapeConv(input_shape=shape, conv_shape=conv_shape, conv_stride=conv_stride,
                                bias_weight=bias_weight)

        self.out_shape = conv.out_shape
        self.detector_shape = (self.out_shape[0], self.out_shape[1])

        self.s_in = InPort(shape=shape)
        self.s_out = OutPort(shape=self.detector_shape)
        self.vth = Var(shape=(1,), init=vth)
        self.detector_vu = Var(shape=(1,), init=detector_dv)
        self.u = Var(shape=(60, 57), init=0)


@implements(proc=Semd2dLayer, protocol=LoihiProtocol)
class Semd2dLayerModel(AbstractSubProcessModel):

    def __init__(self, proc):
        # input layer
        shape = proc.init_args.get("shape", (1, 1))
        conv_shape = proc.init_args.get("conv_shape", (1, 1))
        conv_stride = proc.init_args.get("conv_stride", (1, 1))
        thresh_conv = proc.init_args.get("thresh_conv", 0.5)
        vth = proc.init_args.get("vth", 10)
        detector_dv = proc.init_args.get("detector_dv", 0.05)

        bias_weight = (vth / (conv_shape[0] * conv_shape[1])) / thresh_conv
        # convolution layer with reshape proc
        self.conv = ReshapeConv(input_shape=shape, conv_shape=conv_shape, conv_stride=conv_stride,
                                bias_weight=bias_weight)

        out_shape = self.conv.out_shape
        # connect the input to the convolution
        proc.in_ports.s_in.connect(self.conv.in_ports.s_in)

        # lif layer that collects the value from the convolution layer and fires when enough neuron in the input fire
        self.lif = LIF(shape=out_shape, du=1.0, dv=0.0, vth=10.0)

        # conv to lif
        self.conv.s_out.connect(self.lif.a_in)

        n_directions = 16 #TODO change to process variable
        dirs_list = [
            [3, 0], [3, -1], [2, -2], [1, -3], [0, -3], [-1, -3], [-2, -2], [-3, -1], [-3,0], [-3, 1], [-2,2],
            [-1,3], [0,3], [1,3], [2,2], [3,1]
        ]

        self.detector_list = []

        for dir in dirs_list[:]:
            detector_shape = (out_shape[0] - np.abs(dir[1]), out_shape[1] - np.abs(dir[0]))

            if dir[0] > 0:
                col_del_excit = np.arange(-dir[0], 0)
            else:
                col_del_excit = np.arange(0, -dir[0])

            if dir[1] > 0:
                row_del_excit = np.arange(-dir[1], 0)
            else:
                row_del_excit = np.arange(0, -dir[1])

            if dir[0] > 0:
                col_del_trig = np.arange(0, dir[0])
            else:
                col_del_trig = np.arange(dir[0], 0)

            if dir[1] > 0:
                row_del_trig = np.arange(0, dir[1])
            else:
                row_del_trig = np.arange(dir[1], 0)

            # removes columns and row from the lif to match the dimension of the time difference detector
            # change_dim_excit = ChangeDim(shape=(out_shape), col_del=col_del_excit, row_del=row_del_excit, weight=11.0)
            self.change_dim_excit = ChangeDim(shape=(out_shape), col_del=col_del_excit, row_del=row_del_excit, weight=11.0)
            self.change_dim_trig = ChangeDim(shape=(out_shape), col_del=col_del_trig, row_del=row_del_trig)

            # time difference detector for the left motion
            self.detector = TDE2D(shape=detector_shape, dv=detector_dv, du=0.9)

            self.add_dim = AddDim(in_shape=detector_shape, out_shape=out_shape, dis_x=dir[0], dis_y=dir[1])

            self.detector_list.append(self.detector)

            self.lif.s_out.connect(self.change_dim_excit.s_in)
            self.lif.s_out.connect(self.change_dim_trig.s_in)

            self.change_dim_trig.s_out.connect(self.detector.t_in)
            self.change_dim_excit.s_out.connect(self.detector.a_in)

            self.detector.s_out.connect(self.add_dim.s_in)
            self.add_dim.s_out.connect(proc.out_ports.s_out)

            #proc.vars.u.alias(self.detector.vars.u)
