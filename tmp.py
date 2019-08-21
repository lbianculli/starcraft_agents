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

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

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


# def add_vtarg_and_adv(seg, gamma, lam):  # dont think I need I alreay can calculate these 

def learn(*,
        network,
        env,
        total_timesteps,
        timesteps_per_batch=1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters=3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm
    Parameters:
    ----------
    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets 
    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class
    timesteps_per_batch     timesteps per gradient estimation batch
    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )
    ent_coef                coefficient of policy entropy term in the optimization objective
    cg_iters                number of iterations of conjugate gradient algorithm
    cg_damping              conjugate gradient damping
    vf_stepsize             learning rate for adam optimizer used to optimie value function loss
    vf_iters                number of iterations of value function optimization iterations per each policy optimization step
    total_timesteps           max number of timesteps
    max_episodes            max number of episodes
    max_iters               maximum number of policy optimization iterations
    callback                function to be called with (locals(), globals()) each policy optimization step
    load_path               str, path to load the model from (default: None, i.e. no model is loaded)
    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
    Returns:
    -------
    learnt model
    '''
    if MPI is not None:  # still dont know what mpi is
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0
        
    config=tf.ConfigProto(
                allow_soft_placement=True,
                inter_op_parallelism_threads=cpus_per_worker,
                intra_op_parallelism_threads=cpus_per_worker)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) 
    
    ops_spec = env.obs_spec()
    act_spec = env.act_spec()
    
    #1 get two policies("old" and "new")
    #2 get kl for old_policy - new_policy
    #3 get entropy of new policy
    #4 kl = tf.reduce_mean(kl)
    #5 mean_ent = tf.reduce_mean(entropy)
    #6 entropy = mean_ent * entropy_coef
    #7 get value function (vf)
    #8 vferr = tf.reduce_mean(tf.square(vf - ret))  # in baselines file ret is a placeholder
    #9 ratio = tf.exp(new_policy.logprob(action) - old_policy.logprob(action))  # advantage * pnew / pold
    #10 surrgain = tf.reduce_mean(ratio * atarg)  # atarg is ph for advantage
    
    
    

    
###
    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
