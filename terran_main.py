import sys
import os

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

from pysc2.env import environment
import numpy as np
import deepq_mineral_shards
import datetime


def main():
	agent = TerranAgent()
	map='Simple64'
	with sc2_env.SC2Env(
	    map_name=map,
	    step_mul=8,
	    visualize=True,
	    agent_interface_format=features.AgentInterfaceFormat(
				feature_dimensions=features.Dimensions(screen=16, minimap=16)) as env:
		
		agent.setup(env.observation_spec(), env.action_spec())
		timesteps = env.reset()
		agent.reset()

		while True:
			step_actions = [agent.step(timesteps[0])]
			if timesteps[0].last():  
				break
			timesteps = env.step(step_actions)  
