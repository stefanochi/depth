import sys

sys.path.append("C:/Users/schiavaz/lava-nc/depth/")

from .runner import Runner
from . import flow_utils
import numpy as np
from tqdm import tqdm

from lava.proc.io.source import RingBuffer
from lava.proc.io.sink import RingBuffer as SinkBuffer
from lava.proc.monitor.process import Monitor

from lava.magma.core.run_configs import RunConfig, Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

from semd.proc.semd_2d.process import Semd2dLayer


class LavaRunner(Runner):
    def __init__(self):
        return
