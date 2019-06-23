import numpy as np
import os
import tensorflow as tf
import logging
import pickle
import time

from absl import flags
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule

from dqn_utils import LinearSchedule
from dqn_value_estimator import PlayerRelativeCNN


class BuildMarinesAgent():
    def __init__(self,
                 learning_rate=5e-4,  
                 gamma=.99,
                 epsilon_max=1.0,
                 epsilon_min=0.02,
                 epsilon_decay_steps=300000,
                 learning_starts=1000,
                 train_freq=100,
                 target_update_freq=5000,
                 max_buffer_size=100000,
                 batch_size=16,
                 prioritized_replay_beta_iters = 300000, 
                 training=True,
                 indicate_nonrandom_action=False,
                 prioritized=True,
                 prioritized_alpha = .6,  
                 prioritized_beta = .4,  
                 save_file='C:/Users/lbianculli/venv1/sc_bot/minigames/build_marines/logs/network_saves',
                 save_dir='C:/Users/lbianculli/venv1/sc_bot/minigames/build_marines/logs/ckpts/',
                 ckpt_name='collect_minerals_6-23',
                 summary_path='C:/Users/lbianculli/venv1/sc_bot/minigames/build_marines/logs/summaries/',
                 buffer_path='C:/Users/lbianculli/venv1/sc_bot/minigames/build_marines/logs/buffers/buffer_6-23',
                 logdir='C:/Users/lbianculli/venv1/sc_bot/minigames/build_marines/logs/variable_logs.txt',
                 log=True):

    super(DQNMoveOnlyAgent, self).__init__()

        # NN hparams
        self.learning_rate = learning_rate
        self.gamma = gamma

        # agent hparams
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.indicate_nonrandom_action = indicate_nonrandom_action  # not sure exactly
        self.prioritized = prioritized
        self.prioritized_alpha = prioritized_alpha
        self.prioritized_beta = prioritized_beta
        self.save_file = save_file
        self.batch_size = batch_size
        self.log = log

        # other
        self.training = training  # not sure exactly
        self.max_reward = 0
        self.total_reward = 0
        self.last_state = None
        self.last_action = None

        if self.prioritized:
            self.buffer_file = buffer_path + '_prioritized.p'
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters, initial_p=self.prioritized_beta, final_p=1.0)
        else:
            self.buffer_file = buffer_path + '.p'

        # load and set epsilon
        if os.path.isfile(self.save_file + '.npy'):
            self.epsilon, self.initial_step = np.load(self.save_file + '.npy')  
            print(f'epsilon loaded: {self.epsilon}')
        else:
            self.epsilon = 1.0
            self.initial_step = 0
        self.epsilons = [self.epsilon]


        # for saving and loading files
        if save_dir:
            self.online_save_dir = save_dir + 'online/' 
            self.target_save_dir = save_dir + 'target/'

        if ckpt_name:
            self.ckpt_name = ckpt_name

        if summary_path:
            self.online_summary_path = summary_path + 'online/' 
            self.target_summary_path = summary_path + 'target/'

        if self.log:
            self.init_logger(logdir)

        # build network
        if save_dir and ckpt_name:
            self.online_save_path = self.online_save_dir + ckpt_name + '.ckpt'
            self.target_save_path = self.target_save_dir + ckpt_name + '.ckpt'
        print("Building models...")
        tf.reset_default_graph()
        self.online_network = PlayerRelativeCNN(spatial_dims=FEATURE_SCREEN_SIZE,
                                                 learning_rate=self.learning_rate,
                                                 save_path=self.online_save_path,
                                                 summary_path=self.online_summary_path,
                                                 name='DQN')
        if self.training:
            # set up target_net and initialize replay buffer
            self.target_network = PlayerRelativeCNN(spatial_dims=FEATURE_SCREEN_SIZE,
                                                            learning_rate=self.learning_rate,
                                                            save_path = self.target_save_path,
                                                            summary_path = self.target_summary_path,
                                                            name='target_network')

        # initialize tf session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        print('Initialization complete.')

        # check for and load networks/buffer if possible
        if os.path.isfile(self.online_save_path + '.index') and os.path.isfile(self.target_save_path + '.index'):
            self.online_network.load(self.sess)
            self.target_network.load(self.sess)

        # check for buffer to load
        if os.path.isfile(self.buffer_file):
            with open(self.buffer_file, 'rb') as f:
                self.replay_buffer = pickle.load(f)
        else:
            if self.prioritized:  # alpha = 0 is same as uniform
                self.replay_buffer = PrioritizedReplayBuffer(max_buffer_size, self.prioritized_alpha)
            else:
                self.replay_buffer = ReplayBuffer(max_buffer_size)

        self.online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DQN')
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_network')
        self.online_network._init_train_fn(self.online_vars, grad_norm_clipping=10) 
        self.target_network._init_train_fn(self.target_vars, grad_norm_clipping=10)  # not sure if there are better values for clip

        print('online and target models loaded.')
        self._tf_init()

        if self.training:
            self._update_target_network()  
        else:
            self._tf_init()


    def reset(self):
        self.episodes += 1
        self.reward = 0

        if self.training:
            self.last_state = None
            self.last_action = None
            self.episode = self.online_network.global_episode.eval(self.sess)


    def step(self, obs):
    	
