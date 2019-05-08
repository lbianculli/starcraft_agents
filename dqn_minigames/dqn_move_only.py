"""Deep Q-learning agents."""
import numpy as np
import os
import tensorflow as tf

import agents.networks.value_estimators as nets

from absl import flags
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from openai.baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


FEATURE_SCREEN_SIZE = [32,32]  # should these be len 2?
FEATURE_MINIMAP_SIZE = [32,32]  # not actually needed for minigames
FUNCTIONS = sc2_actions.FUNCTIONS


class DQNMoveOnlyAgent(base_agent.BaseAgent):
    ''' DQN that takes in player_relative features and returns movements '''
    
    def __init__(self, 
                 learning_rate,
                 gamma,
                 epsilon_max, 
                 epsilon_min,
                 epsilon_decay_steps,
                 train_freq,
                 target_update_frequency,
                 indicate_nonrandom_action,
                 max_buffer_size,
                 batch_size, 
                 training,
                 save_dir=None,
                 ckpt_name=None,
                 summary_path=None):
        super(DQNMoveOnly, self).__init__()
        
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
        self.indicate_nonrandom_action = indicate_nonrandom_action  # not sure exactly
        
        # other
        self.training = training  # not sure exactly
        
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
        if self.training:  
            # set up target_net and initialize replay buffer
            self.target_network = PlayerRelativeMovementCNN(self,
                                                            spatial_dims=FEATURE_SCREEN_SIZE,
                                                            learning_rate=self.learning_rate,
                                                            name='target_network')
            self.replay_buffer = ReplayBuffer(max_buffer_size)
            self.batch_size = batch_size
            
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
            
            if self.training:  # calls with min epsilon = None
                # predict and take an action
                x, y, action = self._epsilon_greedy_action(state)
                
                # update online DQN/target network if appropriate
                if (self.steps % self.train_freq == 0) and 
                (len(self.replay_buffer) > self.batch_size):
                    self._train_network()
                    
                if self.steps % self.update_train_freq == 0:
                    self._update_target_networks()
                
                # add to replay buffer
                if self.last_state is not None:
                    self.replay_buffer.add(self.last_state, self.last_action, obs.reward, state)
                
                self.last_state = state
                self.last_action = np.ravel_multi_index((x,y), FEATURE_SCREEN_SIZE)
                
            else:
                x, y, action = self.epsilon_greedy_action(state, self.epsilon_min)
                
            if self.indicate_nonrandom_action and action == 'nonrandom':
                # cosmetic difference between random and Q based actions
                return FUNCTIONS.Attack_screen('now', (x, y))
            
            else:
                return FUNCTIONS.Move_screen('now', (x,y))
        else:
            return FUNCTIONS.select_army('select')

    
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
        ''' get all trainable vars from graph for online and target, zip, then assign from online --> target '''
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DQN') 
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_network')
        
        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))  
            
        self.sess.run(update_op)
        
        
    def _epsilon_greedy_action(self, state, epsilon=None):
        ''' Choose an action from the state with eps greedy *** needs updates *** '''
        step = self.network.global_step.eval(self.sess)  #global_step +1 after the variables have been updated 
        
        if epsilon is None:
            epsilon = max(self.epsilon_min,
                          (self.epsilon_max - ((self.epsilon_max - self.epsilon_min) * step / self.epsilon_decay_steps)))
            
        if epsilon > np.random.rand():
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])
            
            return x, y, random
        
        else:
            inputs = np.expand_dims(state, 0)  # state = obs (from above)
            q_values = self.sess.run(self.network.flatten, feed_dict={self.network.inputs:inputs})  # flatten for unravel
            best_action = np.argmax(q_values)
            x, y = np.unravel_index(max_index, FEATURE_SCREEN_SIZE)  # not entirely sure why this is best, complicated
            
            return x, y, 'nonrandom'
        
        
    def _train_network(self):
        states, actions, targets = self._get_batch()
        self.network.optimizer_op(self.sess, states, actions, targets)
            
            
    def _get_batch(self):
        ''' 
        uses openai.baselines replay buffer to sample transitions.
        returns batch of observations_t, actions_t and targets (rewards) 
        '''
        batch = self.replay_buffer.sample(self.batch_size)  # init requires size, sample requires batch_size
        obses, actions, obses_tp1, rewards, dones = batch  # these will be arrays of len batch_size
        
        # one-hot encode actions
        actions = np.eye(np.prod(FEATURE_SCREEN_SIZE))[actions]  # how does this work?
        
        # get targets
        next_outputs = self.sess.run(
            self.target_network.output(feed_dict={self.target_network.inputs: obses_tp1}))  # why tp1 over current?
        targets = [rewards[i] + self.gamma * np.max(next_output[i]) for i in range(self.batch_size)]
        
        return obses, actions, targets
        

