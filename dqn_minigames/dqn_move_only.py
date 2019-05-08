"""Deep Q-learning agents."""
import numpy as np
import os
import tensorflow as tf

import agents.networks.value_estimators as nets

from absl import flags
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

FEATURE_SCREEN_SIZE = [32,32]
FEATURE_MINIMAP_SIZE = [32,32]
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
        self.network = PlayerRelativeMovementCNN(self,
                                                 spatial_dims=FEATURE_SCREEN_SIZE,
                                                 learning_rate=self.learning_rate,
                                                 save_path=self.save_path,
                                                 summ_path=self.summary_path,
                                                 name='DQN')
        if self.training:  # set up target_net
            self.target_network = PlayerRelativeMovementCNN(self,
                                                            spatial_dims=FEATURE_SCREEN_SIZE,
                                                            learning_rate=self.learning_rate,
                                                            name='target_network')
            
        print('Initialization complete.')
        self.last_state = None
        self.last_action = None
        
        
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
            state = obs.observation.feature_screen.player_relative
            
            if self.training:
                # predict and take an action
                x, y, action = self._epsilon_greedy_action_selection(state)
                
                # update online DQN 
                # ...

    
    def _handle_episode_end(self):
        ''' Save weights and write summaries '''
        self.network.increment_global_episode_op(self.sess)
        
        #save current model and write summaries
        self.network.save_model(self.sess)
        states, actions, targets = self.get_batch()
        self.network.write_summary(self.sess, states, actions, targets, self.reward)
        print('Model saved and summaries written.')
        
    def _tf_init(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        
    def _update_target_networks(self):
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DQN') # what does this do exactly?
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_network')
        
        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))  # what is this ? 
            
        self.sess.run(update_op)
        
        
    def _epsilon_greedy_action(self, state, epsilon=None):
        ''' Choose an action from the state with eps greedy *** needs updates *** '''
        step = self.network.global_step.eval(self.sess)  #global_step +1 after the variables have been updated (opt/loss?)
        
        if epsilon is None:
            epsilon = max(self.epsilon_min,
                          (self.epsilon_max - ((self.epsilon_max - self.epsilon_min) * step / self.epsilon_decay_steps)))
            
        if epsilon > np.random.rand():
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])
            
            return x, y, random
        else:
            inputs = np.expand_dims(state, 0)  # state = obs (from above)
            q_values = self.sess.run(self.network.flatten, feed_dict={self.network.inputs:inputs})  # why need to flatten?
            best_action = np.argmax(q_values)
            x, y = np.unravel_index(max_index, feature_Screen_size)
            return x, y, 'nonrandom'
        
        
    def _train_network(self):
            
        

        
        
                
                
              
            
            
            
            
            
            

