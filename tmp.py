from gailtf.baselines.common import explained_variance, zipsame, dataset, Dataset, fmt_row
from gailtf.baselines import logger
import gailtf.baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time, os
from gailtf.baselines.common import colorize
from mpi4py import MPI
from collections import deque
from gailtf.baselines.common.mpi_adam import MpiAdam
from gailtf.baselines.common.cg import cg
from contextlib import contextmanager
from gailtf.common.statistics import stats
import ipdb
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from pysc2.lib import actions as sc_action
from pysc2.lib import static_data
from pysc2.lib import features
from pysc2.lib import FUNCTIONS
from gym import spaces

from gailtf.baselines.common.mpi_moments import mpi_moments

import math

import gin.tf
import tensorflow as tf

from envs.base import Spec
from utils import StreamLogger
from utils.tensorflow import SessionManager
from utils.typing import ModelBuilder, PolicyType
from agents.base import SyncRunningAgent, ActorCriticAgent, DEFAULTS


# extract_observation() can be replaced by obs_wrapper functionality
"""
1. Collect set of s,a pairs with Q value estimates
2. Construct estimated objective and constraint
3. solve optimization to update theta
"""


@gin.configurable('TRPOAgent')
class TRPOAgent(SyncRunningAgent, ActorCriticAgent):
    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        model_fn: ModelBuilder=None,
        policy_cls: PolicyType=None,
        sess_mgr: SessionManager=None,
        optimizer: tf.train.Optimizer=None,
        n_envs=4,
        value_coef=DEFAULTS['value_coef'],
        entropy_coef=DEFAULTS['entropy_coef'],
        traj_len=DEFAULTS['traj_len'],
        batch_sz=DEFAULTS['batch_sz'],
        discount=DEFAULTS['discount'],
        gae_lambda=DEFAULTS['gae_lambda'],
        clip_rewards=DEFAULTS['clip_rewards'],
        clip_grads_norm=DEFAULTS['clip_grads_norm'],
        normalize_returns=DEFAULTS['normalize_returns'],
        normalize_advantages=DEFAULTS['normalize_advantages'],
    ):
        SyncRunningAgent.__init__(self, n_envs)
        ActorCriticAgent.__init__(self, obs_spec, act_spec, sess_mgr=sess_mgr, **kwargs)
        self.logger = StreamLogger(n_envs=n_envs, log_freq=10, sess_mgr=self.sess_mgr)

    def traj_segment_generator(pi, env, discriminator, horizon, expert_dataset, stochastic):
        """
        steps thru env
        """
        # initialize state vars. not sure which i will end up needing
        t = 0
        new = True
        rew = 0.0
        true_rew = 0.0
        cur_ep_ret = 0
        cur_ep_len = 0
        ep_rets = []
        ep_lens = []

        ob, *_ = env.reset()  # hopefulle these work
        ac = self.get_action(obs)

        # setup history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        pred_vals = np.zeros(horizon, 'float32')
        dones = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prev_acs = acs.copy()

        while True:  #
            prev_ac = ac
            ac, pred_val = self.get_action_and_value(ob)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value --> dont totally get below
            if t > 0 and t % horizon == 0:  # if terminal step
                yield {"ob" : obs, "rew" : rews, "pred_vals" : pred_val, "done" : dones,
                    "ac" : acs, "prev_ac" : prev_acs, "next_pred_val": pred_val * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
                _, pred_val = self.get_action_and_value(ob)
                ep_rets = []
                ep_lens = []

            i = t % horizon  # start filling in history arryas
            obs[i] = ob
            pred_vals[i] = pred_val
            dones[i] = done
            acs[i] = ac
            prev_acs[i] = prev_ac

            obs, rew, done = env.step(ac)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if done:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1






