"""Deep Q-learning agents."""
import numpy as np
import os
import tensorflow as tf

import agents.networks.value_estimators as nets

from absl import flags
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions


FUNCTIONS = sc2_actions.FUNCTIONS

# class memory is at the top, reconcile w/ replay buffer if possible

class DQNMoveOnly(base_agent.BaseAgent):


