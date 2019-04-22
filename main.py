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
            
			model = deepq.models.cnn_to_mlp(  # outputs, kernel size, stride
				convs=[(16, 5, 1), (32, 5, 1)], hiddens=[256], dueling=True)
			
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
	global max_mean_reward, last_filename
	
	if 'done' in locals and locals['done'] == True:
		if('mean_100ep_reward' in locals and locals ['num_episodes'] >= 10
		   and locals['mean_100ep_reward'] > max_mean_reward):
			
			print('mean_100ep_reward: {locals['mean_100ep_reward']}\nmax_mean_reward: {max_mean_reward}')
			
		if not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq/')):
			try:
				os.mkdir(os.path.join(PROJ_DIR, 'models/'))
			except Exception as e:
				try:
					os.mkdir(os.path.join(PROJ_DIR, 'models/deepq/'))
				except Exception as e:
					print(str(e))
		if last_filename != '':
				
				
			

  
  
