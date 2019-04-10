import sys
import os
import datetime
import numpy as np
from absl import flags
import threading
import time
import random
import nsml

from baselines import deepq
from common.vec_env.subproc_vec_env import SubprocVecEnv
from a2c.policies import CnnPolicy
from a2c import a2c
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.env import environment

import deepq_mineral_shards


def main():
