from .base import *
from .random import RandomAgent
from .sc_ac import AdvantageActorCriticAgent
from .ppo import ProximalPolicyOptimizationAgent
from .sl_agent import SupervisedAgent

A2C = AdvantageActorCriticAgent
PPO = ProximalPolicyOptimizationAgent
SL = SupervisedAgent

registry = {
    'a2c': A2C,
    'ppo': PPO,
    'sl': SL
}
