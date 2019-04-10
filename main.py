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
  agent = TerranAgent()
  map = 'Simple64'
  try:
      while True:  
          with sc2_env.SC2Env(
              map_name=map,
              players=[sc2_env.Agent(sc2_env.Race.terran),
                       sc2_env.Bot(sc2_env.Race.terran,
                                   sc2_env.Difficulty.very_easy)],
              agent_interface_format=features.AgentInterfaceFormat( 
                  feature_dimensions=features.Dimensions(screen=84, minimap=64),
                  use_feature_units=True), 
              step_mul=25, 
              game_steps_per_episode=0, 
              visualize=True  # True gives you a full version of the visuals
              ) as env:

              agent.setup(env.observation_spec(), env.action_spec())
              timesteps = env.reset()
              agent.reset()

              while True:
                  step_actions = [agent.step(timesteps[0])]
                  if timesteps[0].last():  
                      break
                  timesteps = env.step(step_actions) 

  
  
