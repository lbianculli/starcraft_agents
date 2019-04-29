import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

# steps (for DQN): create env, create model, act = .learn (this is essentially the agent step), setup callback
# actor critic requires the subproc for env.
# the above steps are for [mineral shards] minigames. What about regular game?
# they use separate agent classes for minigames as well. if current mineral shards works, should be able to copy framework for actual. 


class TerranAgent(base_agent.BaseAgent):
	def __init__(self, ):



def main():
	with sc2_env.SC2Env(
		map_name='Simple64', 
		step_mul=8,
		visualize=True,
		screen_size_px=(64,64),
		minimap_size_px=(64,64)) as env:
						
    
  
  
  
  
  

