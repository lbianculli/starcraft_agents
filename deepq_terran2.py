# https://github.com/chris-chris/pysc2-examples/blob/master/deepq_mineral_shards.py
# i think for this, not gonna use an agent?

import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
from absl import app, flags
import pickle
import pandas as pd
import logging
import math
import random
import zipfile

import baselines.common.tf_util as U
from baselines import logger, deepq
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.models import build_q_func
from baselines.deepq.utils import ObservationInput


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



def load(path, act_params, num_cpu=4):  # much of this copied from openai/baselines
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

    sess = U.make_session(num_cpu)
    sess.__enter__()


    def make_obs_ph(name):
        return U.BatchInput((16, 16), name=name)  # dont know if this will work


    act_x, train_x, update_target_x, debug_x = deepq.build_train(
        make_obs_ph = make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(lr),
        gamma=gamma,
        grad_norm_clipping=10,
        scope='deepq_x')

    act_y, train_y, update_target_y, debug_y = deepq.build_train(
        make_obs_ph = make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(lr),
        gamma=gamma,
        grad_norm_clipping=10,
        scope='deepq_y')

    act_params = {
    'make_obs__ph': make_obs_ph,
    'q_func': q_func,
    'num_actions': num_actions
    }

    # create replay buffer -- does it really make sense to use for both x and y?
    if prioritized_replay:
        replay_buffer_x = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        replay_buffer_y = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)

        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule_x = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)

        beta_schedule_y = LinearSchedule(prioritized_replay_beta_iters,
                                         initial_p=prioritized_replay_beta0,
                                         final_p=1.0)
    else:
        replay_buffer_x = ReplayBuffer(buffer_size)
        replay_buffer_y = ReplayBuffer(buffer_size)

        beta_schedule_x = None
        beta_schedule_y = None

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(max_timesteps * exploration_fraction),
                                          initial_p=1.0,
                                          final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target_x()
    update_target_y()

    episode_rewards = [0.0]
    saved_mean_reward = None
    
    # Now onto the actual SC2 interaction -- everything below this is guesswork thusfar. forget how this works
    obs = env.reset()
    screen = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.NEUTRAL).astype(int)
    player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
    player = [int(player_x.mean()), int(player_y.mean())]
    reset = True

    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join('model/', 'mineral_shards')
        print(model_file)

        # ITERATION STARTS HERE
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # take action and update exploration to newest value
            kwargs = {}
            if not param_noise:  # think all variance reduction stuff
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                if param_noise_threshold >= 0:
                    update_param_noise_threshold = param_noise_threshold
                else:
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(num_actions))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noice_scale'] = True

            # THIS is where it needs to go down, how to choose an actions? Dont I need categorical and spatial actions?
            # the [None] inserts a dimension along axis=0
            action_x = act_x(np.array(screen)[None], update_eps=update_eps, **kwargs)[0]  # takes screen, adds dim, passes to act_
            action_y = act_y(np.array(screen)[None], update_eps=update_eps, **kwargs)[0]  # returns an x and y coord respectively

            reset = False
            coord = [player[0], player[1]]
            r = 0
            coord = [action_x, action_y]  #  why two coord?
            
            #*** got rid of a bunch of minigame specific actions here. Not sure what else needs to be put here/moved around.
            
            r = obs[0].reward
            done = obs[0].step_type == environment.StepType.LAST  # something like this

            replay_buffer_x.add(screen, action_x, r, new_screen, float(done))
            replay_buffer_y.add(screen, action_y, r, new_screen, float(done))

            new_screen = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.NEUTRAL).astype(int)

            episode_rewards[-1] += r
            reward = episode_rewards[-1]

            if done:
                obs = env.reset()
                player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
                screent = (player_relative == _PLAYER_NEUTRAL).astype(int)

                player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
                player = [int(player_x.mean()), int(player_y.mean())]

                # Select all marines first -- do i need?
                env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
                episode_rewards.append(0.0)

                reset = True

            if t > learning_start and t % train_freq == 0:  
                # minimize the error of Bellman equation on a batch sampled from replay buffer
                if prioritized_replay:
                    experience_x = replay_buffer_x.sample(batch_size, beta=beta_schedule_x.value(t))
                    (obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x, weights_x, batch_idxes_x) = experience_x

                    experience_y = replay_buffer_y.sample(batch_size, beta=beta_schedule_y.value(t))
                    (obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y, weights_y, batch_idxes_y) = experience_y

                else:
                    obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x = replay_buffer_x.sample(batch_size)
                    weights_x, batch_idxes_x = np.ones_like(rewards_x), None

                    obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y = replay_buffer_y.sample(batch_size)
                    weights_y, batch_idxes_y = np.ones_like(rewards_y), None

                td_errors_x = train_x(obs_t_x, actions_x, rewards_x, obs_tp1_x, dones_x, weights_x) 
                td_errors_y = train_y(obs_t_y, actions_y, rewards_y, obs_tp1_y, dones_y, weights_y) 

                if prioritized_replay:
                    new_priorities_x = np.abs(td_errors_x) + prioritized_replay_eps
                    new_priorities_y = np.abs(td_errors_x) + prioritizied_replay_eps
                    replay_buffer_x.update_priorities(batch_idxes_x, new_priorities_x)  
                    replay_buffer_y.update_priorities(batch_idxes_y, new_priorities_y) 


            if t > learning_stats and t % target_network_update_freq == 0:  # if its time to update target network
                update_target_x()
                update_target_y()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            
            ### --- lot of logging below --- ###
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("reward", reward)
                logger.reord_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular(f"{int(100*exploration.value(t))} time spent exploring")
                logger.dump_tabular

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                  if print_freq is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                      saved_mean_reward, mean_100ep_reward))
                  U.save_state(model_file)
                  model_saved = True
                  saved_mean_reward = mean_100ep_reward
            if model_saved:
              if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
              U.load_state(model_file)
    
    ### print test act vs. ActWrapper:
    print(f'ActWrapper around act_x: {ActWrapper(act_x)}')
    print(f'act_x: {act_x}')
    return ActWrapper(act_x), ActWrapper(act_y)



######################################################################################
# act: chooses an action based on observation. returns a tensor of shape (BATCH_SIZE,)
# with an action to be performed for every element of the batch.
# train: function that takes transition and optimizes Bellman Error. returns array of shape 
# (batch_size,) with differences between Q and target (TD error) for each element.
######################################################################################


def main():
    map ='Simple64'
    try:
        while True:  # not sure about these args either
            with sc2_env.SC2Env(
                map_name=map,
                step_mul=8,
                visualize=True,
                players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=16, minimap=16), use_feature_units=True),
                game_steps_per_episode=10000) as env:
                    
                    
                    model = deepq.models.cnn_to_mlp(
                          convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)
                    
                    timesteps = env.reset()  
                    act = deepq_mineral_shards.learn(  # could test with max_timesteps=0. that should be able to run at least
                        env,
                        q_func=model,
                        num_actions=16,  # need to change this (most of them probably)
                        lr=5e-4,
                        max_timesteps=200000,
                        buffer_size=25000,
                        exploration_fraction=FLAGS.exploration_fraction,
                        exploration_final_eps=0.01,
                        train_freq=4,
                        learning_starts=10000,
                        target_network_update_freq=1000,
                        gamma=0.99,
                        prioritized_replay=True,
                        callback=deepq_callback)
                        act.save("/home/lbianculli/mineral_shards.pkl")

                        # .reset returns TimeStep object, .step takes an action and returns TimeStep object
                        # ['step_type', 'reward', 'discount', 'observation']  
                        while True:  
                            print(f'step type: {time_steps[0]}, obs: {time_steps[3]}')  # -> step_type vs. observation?
#                             step_actions = [agent.step(timesteps[0])]  # from part_scripted -- this step takes obs
#                             timesteps = env.step(step_actions)  # this step takes action
                            if timesteps[0].last():  # instead of setting up done
                                break
                            # this is the key, just need to find some way to get action in correct format step thru
                            # if unable to do that, is there any way to set this up within the function itself?
                            obs = timesteps[0]  # do i want this below or above timesteps? does it matter?
                            timesteps = env.step(act(obs[None])[0])  # where i think 0 corresponds to the action at step[0] ***
#                             obs, rew, _, _ = env.step()  # dont think i need this                          
                            
                            
    except KeyboardInterrupt:
        pass

if __name__ == 'main':
    main()

