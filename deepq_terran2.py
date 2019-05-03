import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
from absl import app
import pickle
import pandas as pd
import logging
import math
import random
import zipfile

import baselines.common.tf_util as U
import baselines.deepq.utils as U2
from baselines import logger, deepq
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.models import build_q_func
from baselines.deepq.utils import ObservationInput
from deepq_inputs import BatchInput

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units


EPISODE_REWARDS = [0.0]



class ActWrapper(object):
    """ Takes batches of observations and returns actions  """
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

###########################
########## AGENT ##########
###########################

class TerranAgent(base_agent.BaseAgent):
    def __init__(self, # which init does this stuff go into?
                 num_cpu=1,
                 max_timesteps=100000,
                 buffer_size=50000,
                 lr=5e-4,
                 gamma=1.0,
                 scope=None,
                 prioritized_replay=True,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta0=0.4,
                 prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6):

        super(TerranAgent, self).__init__()
        self.num_cpu = num_cpu
        self.max_timesteps = max_timesteps
        self.buffer_size = buffer_size
        self.lr = lr
        self.gamma = gamma
        self.scope = scope
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_eps = prioritized_replay_eps
        self.model = deepq.models.cnn_to_mlp(convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)  # has dueling!!
        self.prev_obs = None
        self.action_x = None
        self.action_y = None
        self.total_reward = 0
        self.step_count = 0


    def step(self, obs, env):
        # print(obs.step_type)
        if self.step_count == 0:  # how to account for last step?
            print('ASDASDASDASDASDSDASD')
            self._setup(len(obs.observation.available_actions))

#         screen = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.NEUTRAL).astype(int)
        self.replay_buffer_x.add(self.prev_obs, self.action_x, self.reward, obs, float(done))  # do i want screen or obs?
        self.replay_buffer_y.add(self.prev_obs, self.action_y, self.reward, obs, float(done))
#         self.replay_buffer_x.add(screen, action_x, step_reward, new_screen, float(done))
#         self.replay_buffer_y.add(screen, action_y, step_reward, new_screen, float(done))

        self.action_x, self.action_y = self.learn(env=env)  # what do these end up looking like? anything else needed?
        step_reward = obs[0].reward
        self.prev_obs = obs

        EPISODE_REWARDS[-1] += step_reward  # supposed to append to episode_rewards if done. best way?
        self.total_reward = EPISODE_REWARDS[-1]

        return actions.FUNCTIONS.no_op()  # make sure runs w/ no_op() before confirming actions


    def make_obs_ph(self, name):  ### HERE ###
          U2.ObservationInput(Box(low=-32.0, high=32.0, shape=(2,)), name=name)


    def _setup(self, num_actions): # how to determine? len(available_actions) prob. should prob be somewhere else
        '''
        initializes session, x/y vars, replay_buffer, beta_schedule, exploration, and target networks
        returns: actwrappers for x and y. SHOULD BE CALLED ONLY ONCE. num actions changes though....
        '''

        self.sess = U.make_session(num_cpu=self.num_cpu)
        self.sess.__enter__()

        self.act_x, self.train_x, self.update_target_x, self.debug_x = deepq.build_train(
            make_obs_ph = self.make_obs_ph,
            q_func=self.model,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(self.lr),
            gamma=self.gamma,
            grad_norm_clipping=10,
            scope='deepq_x')

        self.act_y, self.train_y, self.update_target_y, self.debug_y = deepq.build_train(
            make_obs_ph = self.make_obs_ph,
            q_func=self.model,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(self.lr),
            gamma=self.gamma,
            grad_norm_clipping=10,
            scope='deepq_y')

        self.act_params = {
        'make_obs__ph': make_obs_ph,
        'q_func': self.model,
        'num_actions': num_actions
        }

        if self.prioritized_replay:
            self.replay_buffer_x = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
            self.replay_buffer_y = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)

            if self.prioritized_replay_beta_iters is None:
                self.prioritized_replay_beta_iters = self.max_timesteps
            self.beta_schedule_x = LinearSchedule(self.prioritized_replay_beta_iters,
                                           initial_p=self.prioritized_replay_beta0,
                                           final_p=1.0)

            self.beta_schedule_y = LinearSchedule(prioritized_replay_beta_iters,
                                             initial_p=prioritized_replay_beta0,
                                             final_p=1.0)
        else:
            self.replay_buffer_x = ReplayBuffer(self.buffer_size)
            self.replay_buffer_y = ReplayBuffer(self.buffer_size)

            self.beta_schedule_x = None
            self.beta_schedule_y = None

        self.exploration = LinearSchedule(schedule_timesteps=int(max_timesteps * exploration_fraction),
                                          initial_p=1.0,
                                          final_p=exploration_final_eps)

        U.initialize()
        self.update_target_x()
        self.update_target_y()

#         self.episode_rewards = [0.0]  # do i need/
        self.saved_mean_reward = None

        ### print test ###
        print(f'ActWrapper around act_x: {ActWrapper(act_x)}')
        print(f'act_x: {act_x}')
        return ActWrapper(act_x), ActWrapper(act_y) # do i need the wrapper even?


    def learn(self,
              env,  # will env being a requirement cause any trouble here? could i simply just use within main()
              q_func,
              num_actions,  # ?
              exploration_fraction=.01,
              exploration_final_eps=0.02,
              train_freq=1,
              batch_size=32,
              print_freq=100,
              checkpoint_freq=10000,
              learning_starts=1000,
              gamma=1.0,
              target_network_update_freq=500,
              param_noise=False,
              param_noise_threshold=0.05,
              callback=None,
              load_path=None,
              **network_kwargs
                ):
        '''             ***** MAKE SURE INDENTS ARE CORRECT *****
        performs learning for one step -- tracked by self.step_count -- which consists of
        updating exploration value, picking x/y actions, sampling from and updating buffer,
        updating target networks, and logging.
        '''
        kwargs = {}
        if not param_noise:  # variance reduction stuff (?) could go out of order and check this lecture next
            update_eps = self.exploration.value(self.step_count)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            if param_noise_threshold >= 0:
                update_param_noise_threshold = param_noise_threshold
            else:
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017 for more detail
                update_param_noise_threshold = -np.log(
                    1. - self.exploration.value(self.step_count) + self.exploration.value(self.step_count) / float(num_actions))

        # not sure what this is about
        kwargs['reset'] = reset
        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
        kwargs['update_param_noice_scale'] = True

        # THIS is where it needs to go down, how to choose an actions? Dont I need categorical and spatial actions?
        # the [None] inserts a dimension along axis=0
        # would really help if i knew what act_x/y looked like -- from obs to obs
        action_x = self.act_x(np.array(screen)[None], update_eps=update_eps, **kwargs)[self.step_count]  #*** this is the bottleneck
        action_y = self.act_y(np.array(screen)[None], update_eps=update_eps, **kwargs)[self.step_count]  # return this?
        reset = False
#         coord = [player[0], player[1]]
        coord = [action_x, action_y]  #  why two coord? which one?


        # get transition samples, calculate TD errors, update buffer
        if self.step_count > learning_start and self.step_count % train_freq == 0:
            # minimize the error of Bellman equation on a batch sampled from replay buffer
            if self.prioritized_replay:
                experience_x = self.replay_buffer_x.sample(batch_size, beta=self.beta_schedule_x.value(t))
                experience_y = self.replay_buffer_y.sample(batch_size, beta=self.beta_schedule_y.value(t))

                (obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y, weights_y, batch_idxes_y) = experience_y
                (obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x, weights_x, batch_idxes_x) = experience_x

            else:
                obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x = self.replay_buffer_x.sample(batch_size)
                weights_x, batch_idxes_x = np.ones_like(rewards_x), None

                obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y = self.replay_buffer_y.sample(batch_size)
                weights_y, batch_idxes_y = np.ones_like(rewards_y), None

            td_errors_x = self.train_x(obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x, weights_x)
            td_errors_y = self.train_y(obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y, weights_y)

            if self.prioritized_replay:
                new_priorities_x = np.abs(td_errors_x) + prioritized_replay_eps
                new_priorities_y = np.abs(td_errors_x) + prioritizied_replay_eps
                self.replay_buffer_x.update_priorities(batch_idxes_x, new_priorities_x)
                self.replay_buffer_y.update_priorities(batch_idxes_y, new_priorities_y)

        if self.step_count > learning_starts and self.step_count % target_network_update_freq == 0:
            # update target networks
            self.update_target_x()
            self.update_target_y()

        mean_100ep_reward = round(np.mean(EPISODE_REWARDS[-101:-1]), 1)
        self.num_episodes = len(EPISODE_REWARDS)

        ### --- lot of logging below --- ### does it make sense to do w/ separate method?
#         if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
#             logger.record_tabular("steps", self.step_count)
#             logger.record_tabular("episodes", num_episodes)
#             logger.record_tabular("reward", reward)
#             logger.reord_tabular("mean 100 episode reward", mean_100ep_reward)
#             logger.record_tabular(f"{int(100*exploration.value(t))} time spent exploring")
#             logger.dump_tabular

#         if (checkpoint_freq is not None and t > learning_starts and
#                 num_episodes > 100 and t % checkpoint_freq == 0):
#             if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
#               if print_freq is not None:
#                 logger.log("Saving model due to mean reward increase: {} -> {}".format(
#                   saved_mean_reward, mean_100ep_reward))
#               U.save_state(model_file)
#               model_saved = True
#               saved_mean_reward = mean_100ep_reward
#         if model_saved:
#           if print_freq is not None:
#             logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
#           U.load_state(model_file)
        self.step_count += 1


######################################################################################
# act: chooses an action based on observation. returns a tensor of shape (BATCH_SIZE,)
# with an action to be performed for every element of the batch.
# train: function that takes transition and optimizes Bellman Error. returns array of shape
# (batch_size,) with differences between Q and target (TD error) for each element.
######################################################################################



def main(unused_argv):  # could i incorporate this elsewhere?
    agent = TerranAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name='Blackpink',
                step_mul=8,
                visualize=True,
                players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=32, minimap=32), use_feature_units=True),
                game_steps_per_episode=0
                ) as env:

                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()

                # .reset returns TimeStep object: ['step_type', 'reward', 'discount', 'observation']
                # agent.step takes an obs and returns action. env.step takes an action and returns TimeStep object

                while True:
                    step_actions = [agent.step(obs=timesteps[0], env=env)]  # step w/ obs to get action. can i do env like this?

                    if timesteps[0].last():
                        break

                    EPISODE_REWARDS.append(0.0)
                    timesteps = env.step(step_actions)  # step w/ action


    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)

