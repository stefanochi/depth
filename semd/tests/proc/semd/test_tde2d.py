import unittest
import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.proc.dense.process import Dense

from semd.proc.tde_2d.process import TDE2D
from semd.proc.average_layer.process import AverageLayer
from semd.proc.reshape_conv.process import ReshapeConv
from semd.proc.camera_input.process import CameraInputLayer

from lava.lib.dnf.operations.operations import Convolution


class LifRunConfig(RunConfig):
    """Run configuration selects appropriate LIF ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""

    def __init__(self, custom_sync_domains=None, select_tag='fixed_pt'):
        super().__init__(custom_sync_domains=custom_sync_domains)
        self.select_tag = select_tag

    def select(self, proc, proc_models):
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError("No legal ProcessModel found.")


class VecSendProcess(AbstractProcess):
    """
    Process of a user-defined shape that sends an arbitrary vector

    Parameters
    ----------
    shape: tuple, shape of the process
    vec_to_send: np.ndarray, vector of spike values to send
    send_at_times: np.ndarray, vector bools. Send the `vec_to_send` at times
    when there is a True
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape", (1,))
        vec_to_send = kwargs.pop("vec_to_send")
        send_at_times = kwargs.pop("send_at_times")
        num_steps = kwargs.pop("num_steps", 1)
        self.shape = shape
        self.num_steps = num_steps
        self.vec_to_send = Var(shape=shape, init=vec_to_send)
        self.send_at_times = Var(shape=(num_steps,), init=send_at_times)
        self.s_out = OutPort(shape=shape)


@implements(proc=VecSendProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('floating_pt')
class PyVecSendModelFloat(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, float)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        if self.send_at_times[self.time_step - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


class VecRecvProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_steps = kwargs.get("num_steps", 1)
        shape_in = kwargs.get("shape_in", (1, 1))
        shape = (num_steps, shape_in[0], shape_in[1])
        self.s_in = InPort(shape=shape_in)
        self.spk_data = Var(shape=shape, init=np.zeros(shape))


@implements(proc=VecRecvProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag('floating_pt')
class PySpkRecvModelFloat(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    spk_data: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        spk_in = self.s_in.recv()
        self.spk_data[self.time_step - 1, :] = spk_in


class TestTd2d(unittest.TestCase):

    def test_xy_positive(self):
        """
        Test the time difference output when both x and y
        are moving in the positive direction (up and to the right)
        """
        shape = (3, 3)
        num_steps = 10

        td2d = TDE2D(shape=shape)

        up_times = np.zeros((num_steps,), dtype=bool)
        up_times[2] = True
        up_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=up_times)
        down_times = np.zeros((num_steps,), dtype=bool)
        down_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=down_times)
        right_times = np.zeros((num_steps,), dtype=bool)
        right_times[4] = True
        right_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                   vec_to_send=np.ones(shape, dtype=float),
                                   send_at_times=right_times)

        left_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=np.zeros(num_steps))

        trig_times = np.zeros((num_steps,), dtype=bool)
        trig_times[5] = True
        trig_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=trig_times)

        tx_times = np.ones((num_steps,), dtype=bool)
        tx_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=tx_times)

        ty_times = np.ones((num_steps,), dtype=bool)
        ty_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=ty_times)

        u_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)
        v_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)

        up_snd.s_out.connect(td2d.up_in)
        down_snd.s_out.connect(td2d.down_in)
        left_snd.s_out.connect(td2d.left_in)
        right_snd.s_out.connect(td2d.right_in)

        tx_snd.s_out.connect(td2d.t_u)
        ty_snd.s_out.connect(td2d.t_v)

        trig_snd.s_out.connect(td2d.trig_in)

        td2d.u_out.connect(u_rcv.s_in)
        td2d.v_out.connect(v_rcv.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        td2d.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        u_data = u_rcv.spk_data.get()
        v_data = v_rcv.spk_data.get()
        td2d.stop()

        expected_out_v = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_v[5, :] = 3.

        expected_out_u = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_u[5, :] = 1.

        self.assertTrue(np.all(expected_out_v == v_data), msg="{0}".format(v_data))
        self.assertTrue(np.all(expected_out_u == u_data), msg="{0}".format(u_data))

    def test_xy_negative(self):
        """
        Test the time difference output when both x and y
        are moving in the negative direction (down and to the left)
        """
        shape = (3, 3)
        num_steps = 10

        td2d = TDE2D(shape=shape)

        up_times = np.zeros((num_steps,), dtype=bool)
        up_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=up_times)
        down_times = np.zeros((num_steps,), dtype=bool)
        down_times[2] = True
        down_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=down_times)
        right_times = np.zeros((num_steps,), dtype=bool)
        right_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                   vec_to_send=np.ones(shape, dtype=float),
                                   send_at_times=right_times)
        left_times = np.zeros((num_steps,))
        left_times[4] = True
        left_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=left_times)

        trig_times = np.zeros((num_steps,), dtype=bool)
        trig_times[5] = True
        trig_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=trig_times)

        tx_times = np.ones((num_steps,), dtype=bool)
        tx_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=tx_times)

        ty_times = np.ones((num_steps,), dtype=bool)
        ty_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=ty_times)

        u_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)
        v_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)

        up_snd.s_out.connect(td2d.up_in)
        down_snd.s_out.connect(td2d.down_in)
        left_snd.s_out.connect(td2d.left_in)
        right_snd.s_out.connect(td2d.right_in)

        tx_snd.s_out.connect(td2d.t_u)
        ty_snd.s_out.connect(td2d.t_v)

        trig_snd.s_out.connect(td2d.trig_in)

        td2d.u_out.connect(u_rcv.s_in)
        td2d.v_out.connect(v_rcv.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        td2d.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        u_data = u_rcv.spk_data.get()
        v_data = v_rcv.spk_data.get()
        td2d.stop()

        expected_out_v = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_v[5, :] = -3.

        expected_out_u = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_u[5, :] = -1.

        self.assertTrue(np.all(expected_out_v == v_data), msg="{0}".format(v_data))
        self.assertTrue(np.all(expected_out_u == u_data), msg="{0}".format(u_data))

    def test_trig_before(self):
        """
        Test the time difference output when the trigger
        is received before the any other spike or at the same timestep.
        If the two spikes arrive at the same timestep, the counter should not start
        """
        shape = (3, 3)
        num_steps = 10

        td2d = TDE2D(shape=shape)

        up_times = np.zeros((num_steps,), dtype=bool)
        up_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=up_times)
        down_times = np.zeros((num_steps,), dtype=bool)
        down_times[2] = True
        down_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=down_times)
        right_times = np.zeros((num_steps,), dtype=bool)
        right_times[2] = True
        right_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                   vec_to_send=np.ones(shape, dtype=float),
                                   send_at_times=right_times)
        left_times = np.zeros((num_steps,))
        left_times[2] = True
        left_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=left_times)

        trig_times = np.zeros((num_steps,), dtype=bool)
        trig_times[2] = True
        trig_times[5] = True
        trig_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=trig_times)

        tx_times = np.ones((num_steps,), dtype=bool)
        tx_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=tx_times)

        ty_times = np.ones((num_steps,), dtype=bool)
        ty_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=ty_times)

        u_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)
        v_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)

        up_snd.s_out.connect(td2d.up_in)
        down_snd.s_out.connect(td2d.down_in)
        left_snd.s_out.connect(td2d.left_in)
        right_snd.s_out.connect(td2d.right_in)

        tx_snd.s_out.connect(td2d.t_u)
        ty_snd.s_out.connect(td2d.t_v)

        trig_snd.s_out.connect(td2d.trig_in)

        td2d.u_out.connect(u_rcv.s_in)
        td2d.v_out.connect(v_rcv.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        td2d.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        u_data = u_rcv.spk_data.get()
        v_data = v_rcv.spk_data.get()
        td2d.stop()

        expected_out_v = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_v[5, :] = 0.

        expected_out_u = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_u[5, :] = 0.

        self.assertTrue(np.all(expected_out_v == v_data), msg="{0}".format(v_data))
        self.assertTrue(np.all(expected_out_u == u_data), msg="{0}".format(u_data))

    def test_negative_events(self):
        """
        Test the output of the network when the input spikes are negative.
        The sign of the input should not influence the sign of the output
        """
        shape = (3, 3)
        num_steps = 10

        td2d = TDE2D(shape=shape)

        up_times = np.zeros((num_steps,), dtype=bool)
        up_times[2] = True
        up_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.full(shape, -1.0, dtype=float),
                                send_at_times=up_times)
        down_times = np.zeros((num_steps,), dtype=bool)
        down_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=down_times)
        right_times = np.zeros((num_steps,), dtype=bool)
        right_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                   vec_to_send=np.ones(shape, dtype=float),
                                   send_at_times=right_times)
        left_times = np.zeros((num_steps,))
        left_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=left_times)

        trig_times = np.zeros((num_steps,), dtype=bool)
        trig_times[4] = True
        trig_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.full(shape, -1.0, dtype=float),
                                  send_at_times=trig_times)

        tx_times = np.ones((num_steps,), dtype=bool)
        tx_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=tx_times)

        ty_times = np.ones((num_steps,), dtype=bool)
        ty_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=ty_times)

        u_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)
        v_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)

        up_snd.s_out.connect(td2d.up_in)
        down_snd.s_out.connect(td2d.down_in)
        left_snd.s_out.connect(td2d.left_in)
        right_snd.s_out.connect(td2d.right_in)

        tx_snd.s_out.connect(td2d.t_u)
        ty_snd.s_out.connect(td2d.t_v)

        trig_snd.s_out.connect(td2d.trig_in)

        td2d.u_out.connect(u_rcv.s_in)
        td2d.v_out.connect(v_rcv.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        td2d.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        u_data = u_rcv.spk_data.get()
        v_data = v_rcv.spk_data.get()
        td2d.stop()

        expected_out_v = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_v[4, :] = 2.

        expected_out_u = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_u[5, :] = 0.

        self.assertTrue(np.all(expected_out_v == v_data), msg="{0}".format(v_data))
        self.assertTrue(np.all(expected_out_u == u_data), msg="{0}".format(u_data))

    def test_different_polarity(self):
        """
        Test the output of the network when the polarity of the
        direction input is different from the polarity of the trigger.
        The counter should  be stopped if the polarity is teh same, but
        it should keep going the the polarity is different
        """
        shape = (3, 3)
        num_steps = 10

        td2d = TDE2D(shape=shape)

        up_times = np.zeros((num_steps,), dtype=bool)
        up_times[2] = True
        up_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.full(shape, -1.0, dtype=float),
                                send_at_times=up_times)
        down_times = np.zeros((num_steps,), dtype=bool)
        down_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=down_times)
        right_times = np.zeros((num_steps,), dtype=bool)
        right_times[2] = True
        right_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                   vec_to_send=np.ones(shape, dtype=float),
                                   send_at_times=right_times)
        left_times = np.zeros((num_steps,))
        left_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=left_times)

        trig_times = np.zeros((num_steps,), dtype=bool)
        trig_times[4] = True
        trig_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.full(shape, -1.0, dtype=float),
                                  send_at_times=trig_times)
        trig_times_pos = np.zeros((num_steps,), dtype=bool)
        trig_times_pos[6] = True
        trig_snd_pos = VecSendProcess(shape=shape, num_steps=num_steps,
                                      vec_to_send=np.full(shape, 1.0, dtype=float),
                                      send_at_times=trig_times_pos)

        tx_times = np.ones((num_steps,), dtype=bool)
        tx_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=tx_times)

        ty_times = np.ones((num_steps,), dtype=bool)
        ty_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=ty_times)

        u_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)
        v_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)

        up_snd.s_out.connect(td2d.up_in)
        down_snd.s_out.connect(td2d.down_in)
        left_snd.s_out.connect(td2d.left_in)
        right_snd.s_out.connect(td2d.right_in)

        tx_snd.s_out.connect(td2d.t_u)
        ty_snd.s_out.connect(td2d.t_v)

        trig_snd.s_out.connect(td2d.trig_in)
        trig_snd_pos.s_out.connect(td2d.trig_in)

        td2d.u_out.connect(u_rcv.s_in)
        td2d.v_out.connect(v_rcv.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        td2d.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        u_data = u_rcv.spk_data.get()
        v_data = v_rcv.spk_data.get()
        td2d.stop()

        expected_out_v = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_v[4, :] = 2.

        expected_out_u = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_u[6, :] = 4.

        self.assertTrue(np.all(expected_out_v == v_data), msg="{0}".format(v_data))
        self.assertTrue(np.all(expected_out_u == u_data), msg="{0}".format(u_data))

    def test_multiple_inputs(self):
        """
        Test the output of the network when multiple inputs are received
        before the trigger. The counter should be reset when the new input arrives.
        The sign should also change if it is different
        """
        shape = (3, 3)
        num_steps = 10

        td2d = TDE2D(shape=shape)

        up_times = np.zeros((num_steps,), dtype=bool)
        up_times[2] = True
        up_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.full(shape, 1.0, dtype=float),
                                send_at_times=up_times)
        down_times = np.zeros((num_steps,), dtype=bool)
        down_times[4] = True
        down_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.ones(shape, dtype=float),
                                  send_at_times=down_times)
        right_times = np.zeros((num_steps,), dtype=bool)
        right_times[4] = True
        right_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                   vec_to_send=np.full(shape, -1.0, dtype=float),
                                   send_at_times=right_times)
        left_times = np.zeros((num_steps,))
        left_times[2] = True
        left_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.full(shape, 1.0, dtype=float),
                                  send_at_times=left_times)

        trig_times = np.zeros((num_steps,), dtype=bool)
        trig_times[7] = True
        trig_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.full(shape, -1.0, dtype=float),
                                  send_at_times=trig_times)
        trig_times_pos = np.zeros((num_steps,), dtype=bool)
        trig_times_pos[6] = True
        trig_snd_pos = VecSendProcess(shape=shape, num_steps=num_steps,
                                      vec_to_send=np.full(shape, 1.0, dtype=float),
                                      send_at_times=trig_times_pos)

        tx_times = np.ones((num_steps,), dtype=bool)
        tx_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=tx_times)

        ty_times = np.ones((num_steps,), dtype=bool)
        ty_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                vec_to_send=np.ones(shape, dtype=float),
                                send_at_times=ty_times)

        u_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)
        v_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)

        # up_snd.s_out.connect(td2d.up_in)
        down_snd.s_out.connect(td2d.down_in)
        # left_snd.s_out.connect(td2d.left_in)
        # right_snd.s_out.connect(td2d.right_in)

        tx_snd.s_out.connect(td2d.t_u)
        ty_snd.s_out.connect(td2d.t_v)

        trig_snd.s_out.connect(td2d.trig_in)
        trig_snd_pos.s_out.connect(td2d.trig_in)

        td2d.u_out.connect(u_rcv.s_in)
        td2d.v_out.connect(v_rcv.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        td2d.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        u_data = u_rcv.spk_data.get()
        v_data = v_rcv.spk_data.get()
        td2d.stop()

        expected_out_v = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_v[6, :] = -2.

        expected_out_u = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_u[7, :] = 3.

        self.assertTrue(np.all(expected_out_v == v_data), msg="{0}".format(v_data))
        self.assertTrue(np.all(expected_out_u == u_data), msg="{0}".format(u_data))


class TestAverageLayer(unittest.TestCase):
    def test_avg_conv(self):
        """
        Test the time difference output when both x and y
        are moving in the positive direction (up and to the right)
        """
        shape = (9, 9)
        num_steps = 10

        avg_layer = AverageLayer(shape=shape, mean_thr=0.2)

        two_times = np.zeros((num_steps,), dtype=bool)
        two_times[2] = True
        vec = np.zeros(shape)
        vec[0, 1] = 2.0
        # vec[2, 2] = 4.0
        two_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                 vec_to_send=vec,
                                 send_at_times=two_times)
        four_times = np.zeros((num_steps,), dtype=bool)
        four_times[3] = True
        vec = np.zeros(shape)
        vec[2, 1] = 4.0
        # vec[2, 2] = 4.0
        four_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=vec,
                                  send_at_times=four_times)

        trig_times = np.zeros((num_steps,), dtype=bool)
        trig_times[4] = True
        trig_snd = VecSendProcess(shape=shape, num_steps=num_steps,
                                  vec_to_send=np.full(shape, 3.0, dtype=float),
                                  send_at_times=trig_times)

        u_rcv = VecRecvProcess(shape_in=shape, num_steps=num_steps)

        two_snd.s_out.connect(avg_layer.trig_in)
        four_snd.s_out.connect(avg_layer.trig_in)
        trig_snd.s_out.connect(avg_layer.trig_in)

        n_elems = shape[0] * shape[1]
        conn_shape = (n_elems, n_elems)

        conv = Convolution(np.full((3, 3), 1.0))
        conv.configure(shape)
        conv_weights = conv._compute_weights() - np.eye(n_elems)

        conv_dense = Dense(shape=conn_shape, weights=conv_weights, use_graded_spike=True)
        n_conv_dense = Dense(shape=conn_shape, weights=conv_weights, use_graded_spike=True)

        avg_layer.s_out.flatten().connect(conv_dense.s_in)
        conv_dense.a_out.reshape(shape).connect(avg_layer.s_in)

        avg_layer.n_out.flatten().connect(n_conv_dense.s_in)
        n_conv_dense.a_out.reshape(shape).connect(avg_layer.n_in)

        avg_layer.s_out.connect(u_rcv.s_in)

        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        avg_layer.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        u_data = u_rcv.spk_data.get()
        avg_layer.stop()

        expected_out_u = np.zeros((num_steps, shape[0], shape[1]))
        expected_out_u[2, 0, 1] = 2.0
        expected_out_u[3, 2, 1] = 4.0
        expected_out_u[4, :] = 3.0

        self.assertTrue(np.all(expected_out_u == u_data), msg="{0}".format(u_data))


class TestCameraInput(unittest.TestCase):
    def test_flow_xy(self):
        vel_vec = np.array([1, -1, 0])
        vel_snd = VecSendProcess(shape=(3,), num_steps=3,
                                 vec_to_send=vel_vec,
                                 send_at_times=[0, 1, 0])

        cam_input = CameraInputLayer(shape=(3, 3),
                                     focal_length=100,
                                     center_x=1,
                                     center_y=1)

        x_rcv = VecRecvProcess(shape_in=(3, 3), num_steps=3)
        y_rcv = VecRecvProcess(shape_in=(3, 3), num_steps=3)

        vel_snd.s_out.connect(cam_input.s_in)

        cam_input.x_out.connect(x_rcv.s_in)
        cam_input.y_out.connect(y_rcv.s_in)

        rcnd = RunSteps(num_steps=3)
        rcfg = LifRunConfig(select_tag='floating_pt')
        cam_input.run(condition=rcnd, run_cfg=rcfg)

        x_data = x_rcv.spk_data.get()
        y_data = y_rcv.spk_data.get()

        cam_input.stop()

        flow = get_translational_flow(vel_vec, 100, [1, 1], (3, 3))
        expected_output_x = np.zeros((3, 3, 3))
        expected_output_x[1][:] = flow[0]
        expected_output_y = np.zeros((3, 3, 3))
        expected_output_y[1][:] = flow[1]

        self.assertTrue(np.all(expected_output_x == x_data), msg="{}".format(x_data))
        self.assertTrue(np.all(expected_output_y == y_data), msg="{}".format(y_data))

    def test_flow_z(self):
        vel_vec = np.array([0, 0, 1])
        vel_snd = VecSendProcess(shape=(3,), num_steps=3,
                                 vec_to_send=vel_vec,
                                 send_at_times=[0, 1, 0])

        cam_input = CameraInputLayer(shape=(3, 3),
                                     focal_length=100,
                                     center_x=1,
                                     center_y=1)

        x_rcv = VecRecvProcess(shape_in=(3, 3), num_steps=3)
        y_rcv = VecRecvProcess(shape_in=(3, 3), num_steps=3)

        vel_snd.s_out.connect(cam_input.s_in)

        cam_input.x_out.connect(x_rcv.s_in)
        cam_input.y_out.connect(y_rcv.s_in)

        rcnd = RunSteps(num_steps=3)
        rcfg = LifRunConfig(select_tag='floating_pt')
        cam_input.run(condition=rcnd, run_cfg=rcfg)

        x_data = x_rcv.spk_data.get()
        y_data = y_rcv.spk_data.get()

        cam_input.stop()

        flow = get_translational_flow(vel_vec, 100, [1, 1], (3, 3))
        expected_output_x = np.zeros((3, 3, 3))
        expected_output_x[1][:] = flow[0]
        expected_output_y = np.zeros((3, 3, 3))
        expected_output_y[1][:] = flow[1]

        self.assertTrue(np.all(expected_output_x == x_data), msg="{}".format(x_data))
        self.assertTrue(np.all(expected_output_y == y_data), msg="{}".format(y_data))

    def test_flow_xyz(self):
        vel_vec = np.array([1, 2, 3])
        vel_snd = VecSendProcess(shape=(3,), num_steps=3,
                                 vec_to_send=vel_vec,
                                 send_at_times=[0, 1, 0])

        cam_input = CameraInputLayer(shape=(3, 3),
                                     focal_length=100,
                                     center_x=1,
                                     center_y=1)

        x_rcv = VecRecvProcess(shape_in=(3, 3), num_steps=3)
        y_rcv = VecRecvProcess(shape_in=(3, 3), num_steps=3)

        vel_snd.s_out.connect(cam_input.s_in)

        cam_input.x_out.connect(x_rcv.s_in)
        cam_input.y_out.connect(y_rcv.s_in)

        rcnd = RunSteps(num_steps=3)
        rcfg = LifRunConfig(select_tag='floating_pt')
        cam_input.run(condition=rcnd, run_cfg=rcfg)

        x_data = x_rcv.spk_data.get()
        y_data = y_rcv.spk_data.get()

        cam_input.stop()

        flow = get_translational_flow(vel_vec, 100, [1, 1], (3, 3))
        expected_output_x = np.zeros((3, 3, 3))
        expected_output_x[1][:] = flow[0]
        expected_output_y = np.zeros((3, 3, 3))
        expected_output_y[1][:] = flow[1]

        self.assertTrue(np.all(expected_output_x == x_data), msg="{}".format(x_data))
        self.assertTrue(np.all(expected_output_y == y_data), msg="{}".format(y_data))

def get_translational_flow(t, f, C, shape):
    u_flow = np.zeros(shape)
    v_flow = np.zeros(shape)
    for x in range(u_flow.shape[1]):
        for y in range(u_flow.shape[0]):
            # shift coordinates to be centered
            xi = x - C[0]
            yi = np.abs(y - u_flow.shape[0] +1) - C[1]

            # compute image flow
            m = np.array([
                [-f, 0, xi],
                [0, -f, yi]
            ])
            r = m @ t

            u_flow[y, x] = r[0]  # x flow
            v_flow[y, x] = r[1]  # y flow

    return u_flow, v_flow

if __name__ == '__main__':
    unittest.main()
