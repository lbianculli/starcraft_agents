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

import mineral_shards_agent


def main(unusedargv):
  # started to figure out whats going on here but a lot to chew on
  map = 'CollectMineralShards'
  try:
      while True:  
        with sc2_env.SC2Env(
            map_name="CollectMineralShards",
            step_mul=step_mul,
            visualize=True,
            screen_size_px=(16, 16),
            minimap_size_px=(16, 16)) as env:
            
			model = deepq.models.cnn_to_mlp(  # make sure specs right
				convs=[(16,8,4), (32, 4, 2)], hiddens=[256], dueling=True)
			
			act = mineral_shards_agent.learn(
				env, 
				q_func=model,
				num_actions=16,  # where this from?
				lr=FLAGS.lr,
				max_timesteps=FLAGS.timesteps,
			  	buffer_size=10000,
				exploration_fraction=FLAGS.exploration_fraction,
				exploration_final_eps=0.01,
				train_freq=4,
				learning_starts=10000,
				target_network_update_freq=1000,
				gamma=0.99,
				prioritized_replay=True,
				callback=deepq_callback)
# 			act.save("mineral_shards.pkl")
				

def deepq_callback(locals, globals): 
				
				
				
			

  
  
