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
    ''' DQN that takes in player_relative features and returns movements '''
    
    def __init__(self, 
                 learning_rate,
                 gamma,
                 epsilon_max, 
                 epsilon_min,
                 epsilon_decay_steps,
                 train_freq,
                 target_update_frequency,
                 max_buffer_size,
                 batch_size, 
                 training,
                 save_dir=None,
                 ckpt_name=None,
                 summary_path=None):
        super(DQNMoveOnly, self).__init__()
        # skipping some for now, make sure to finish later. can also look at baselines for default values
        
        # saving and writing
        if save_dir:
            self.save_dir = save_dir
        if ckpt_name:
            self.ckpt_name = ckpt_name
        if summary_path:
            self.summary_path = summary_path
            
        # NN hparams
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # agent hparams
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        
        # other
        self.training = training
        
        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"
        print("Building models...")
        tf.reset_default_graph()   
#         self.network = 

        
    def reset(self):
        ''' reset episode '''
        self.episodes += 1
        self.reward = 0
        
        if self.training:
            self.last_state = None
            self.last_action = None
            
            episode = self.network.global_episode.eval(self.sess)
            print(f'Global episode number: {episode+1}')

    def step(self, obs):
        ''' If no units selected, selects army. Otherwise, move. '''
        self.steps += 1
        self.reward += obs.reward
        
        # handle terminal steps:
        if self.training and obs.last():
            self._handle_episode_end()
            
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            

