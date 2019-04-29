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
	def __init__(self):  # inherits setup, step, reset as below. main changes go to step
		super(TerranAgent, self).__init__()
	
	def setup(self, obs_spec, action_spec):
		self.obs_spec = obs_spec
		self.action_spec = action_spec

	def reset(self):
		self.episodes += 1

	def step(self, obs):
		self.steps += 1
		self.reward += obs.reward
		
		
		return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])





def main():
	with sc2_env.SC2Env(
		map_name='Simple64', 
		step_mul=8,
		visualize=True,
		screen_size_px=(64,64),
		minimap_size_px=(64,64)) as env:
						
    
  
  
  
  
  

