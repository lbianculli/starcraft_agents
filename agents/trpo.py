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


### adding below here ###


def learn(env, policy_func, *,
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        sample_stochastic=True, task="train",
        ckpt_dir=None, save_per_iter=100,
        load_model_path=None, task_name=None,
        max_sample_traj=1500
        ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    # ob_space = (5*60*60, 10*60*60 , 11 , 524) # minimap, screen, info, available_actions
    # ac_space = (524, 2) # actions argument

    # pi = policy_func("pi", ob_space, ac_space)
    # oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    # Note: these steps looked different when looking at from Github, I would assume that's newer
    #1. setup phs including obs and ac
    #2. get multiple policies: "new" and "old"
    #3. get KLd of old minus new
    #4. get entropy of new policy
    #5. take means of kl and entropy
    #6. entropy_bonus = ent_coeff * mean_ent
    #7. vferr = tf.reduce_mean(tf.square(new_pi.vpred - ret))
    #8. ratio = tf.exp(new_pi.logprob(act) - old_pi.logprob(act))  # is this the ratio for the TRPO optimization?
    #9. surrgain = tf.reduce_mean(ratio * adv_targ)  # surrogate from conservative policy iteration
    #10. optimgain = surrgain + entropy_bonus
    #11. losses = [optimgain, meankl, entbonus, surrgain, meanent]  ...
