import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile

from absl import flags

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.models import build_q_func
from baselines.deepq.utils import ObservationInput

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions


class ActWrapper(object):
    """ Takes batches of observations and returns actions (copied almost exactly from openai/baselines) """
    def __init__(self, act):
        self._act = act


    @staticmethod
    def load(path, act_params, num_cpu=4):
        """ """
        with open(path, 'rb') as f:
            model_data = dill.load(f)
        act = deepq.build_graph.build_act(**act_params)
        sess = U.make_session(num_cpu)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, 'packed.zip')
            with open(arc_path, 'wb') as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, 'model'))

        return ActWrapper(act)


    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)


    def save(self, path):
        """ pickle model to path """
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, 'model'))
            arc_name = os.path.join(td, 'packed.zip')
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))

            with open(arc_name, 'rb') as f:
                model_data = f.read()
            with open(path, 'wb') as f:
                dill.dump((model_data), f)



def load(path, act_params, num_cpu=4):
  """Load act function that was returned by learn function.
  Parameters
  ----------
  path: str
      path to the act function file
  num_cpu: int
      number of cpus to use for executing the policy
  Returns
  -------
  act: ActWrapper
      function that takes a batch of observations
      and returns actions.
  """
  return ActWrapper.load(path, act_params=act_params, num_cpu=num_cpu)



def learn(env,
          q_func,
          num_actions=4,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=.01,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
            ):
      """Train a deepq model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.
    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    sess = U.make_session(num_cpu)
    sess.__enter__()


    def make_obs_ph(name):
        return U.BatchInput((16, 16), name=name)  # dont know if this will work


    act_x, train_x, update_target_x, debug_x = deepq.build_graph.build_train(  # if these dont work, try w/o build_graph
        make_obs_ph = make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(lr),
        gamma=gamma,
        grad_norm_clipping=10,
        score='deepq_x')

    act_y, train_y, update_target_y, debug_y = deepq.build_graph.build_train(
        make_obs_ph = make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(lr),
        gamma=gamma,
        grad_norm_clipping=10,
        score='deepq_y')

    act_params = {
    'make_obs__ph': make_obs_ph,
    'q_func': q_func,
    'num_actions': num_actions
    }


    # create replay buffer (again copied from openai/baselines)
    

  
