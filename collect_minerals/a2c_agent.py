import tensorflow as tf
from collections import deque
import logging
import os
import pickle

import a2c_net as nets
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


# agent interface settings (defaults specified in pysc2.bin.agent)
SCREEN_DIMS = [32, 32]
MINIMAP_DIMS = [32, 32]

# pysc2 convenience
FUNCTIONS = sc2_actions.FUNCTIONS
FUNCTION_TYPES = sc2_actions.FUNCTION_TYPES
FunctionCall = sc2_actions.FunctionCall

# manually state the argument types which take points on screen/minimap
SCREEN_TYPES = [sc2_actions.TYPES[0], sc2_actions.TYPES[2]]
MINIMAP_TYPES = [sc2_actions.TYPES[1]]


class a2cAgent(base_agent.BaseAgent):
    def __init__(self,
                 learning_rate=1e-5,
                 value_gradient_strength=.5,  # .01 for sure not sure about first
                 regularization_strength=.01,
                 gamma=.99,
                 cut_trajectory_steps=40,
                 training=True,
                 save_dir='C:/Users/lbianculli/venv1/sc_bot/minigames/collect_minerals/logs/ckpts/',
                 save_file='C:/Users/lbianculli/venv1/sc_bot/minigames/collect_minerals/logs/network_saves',
                 ckpt_name='collect_minerals_6-29',
                 summary_path='C:/Users/lbianculli/venv1/sc_bot/minigames/collect_minerals/logs/summaries/',
                 logdir='C:/Users/lbianculli/venv1/sc_bot/minigames/collect_minerals/logs/variable_logs.txt',
                 logdir2='C:/Users/lbianculli/venv1/sc_bot/minigames/collect_minerals/logs/variable_logs2.txt',
                 log=True):
        super(a2cAgent, self).__init__()  # what does this do again?

        if save_dir:  # dont need all these
            self.save_dir = save_dir
        if ckpt_name:
            self.ckpt_name = ckpt_name
        if summary_path:
            self.summary_path = summary_path

        self.learning_rate = learning_rate
        self.value_gradient_strength = value_gradient_strength
        self.regularization_strength = regularization_strength
        self.gamma = gamma
        self.cut_trajectory_steps = cut_trajectory_steps
        self.training = training
        self.save_file = save_file

        if log:
            self._init_logger(logdir, logdir2)
            self.logger.info(f'To view Tensorboard, run tensorboard --logdir={summary_path}')

        self.save_path = save_dir + ckpt_name + '.ckpt'

        if os.path.isfile(self.save_file + '.npy'):
            self.initial_steps = np.load(self.save_file + '.npy')  # can i just use loaded step for epsilon as well?
        else:
            self.initial_steps = 0

        tf.reset_default_graph()
        self.network = nets.AlphaCNN(
            screen_dims=SCREEN_DIMS,
            minimap_dims=MINIMAP_DIMS,
            learning_rate=self.learning_rate,
            value_gradient_strength=self.value_gradient_strength,
            regularization_strength=self.regularization_strength,
            save_path=self.save_path,
            summary_path=self.summary_path)
        self.logger.info('Network initialization complete.')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        if os.path.isfile(self.save_path + '.index'):
            self.network.load(self.sess)
            self.logger.info('Network Loaded')
        else:
            self._init_op()

        self.last_state = None
        self.episodes = 0
        self.all_rewards = []

        self.online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ac_net')

    def reset(self):
        # reset isnt currently running.
        self.episodes += 1
        self.steps = 0
        self.reward = 0

        if self.training:
            self.last_action = None
            self.state_buffer = deque(maxlen=self.cut_trajectory_steps)
            self.action_buffer = deque(maxlen=self.cut_trajectory_steps)
            self.reward_buffer = deque(maxlen=self.cut_trajectory_steps)
            self.global_episode = self.network.global_episode.eval(self.sess)


    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward

        if self.training and obs.step_type == 2:
            self._handle_episode_end()

        observation = obs.observation
        screen_features = observation.feature_screen
        minimap_features = observation.feature_minimap
        flat_features = observation.player  
        available_actions = observation.available_actions

        # sample action (function ID, args, arg_types) from policy
        action_id, args, arg_types = self._sample_action(screen_features,
                                                         minimap_features,
                                                         flat_features,
                                                         available_actions)

        if self.training:
            if self.last_action:
                self.state_buffer.appendleft((screen_features, minimap_features, flat_features))
                self.action_buffer.appendleft(self.last_action)
                self.reward_buffer.appendleft(obs.reward)

            # train if past certain amount of steps
            if self.steps % self.cut_trajectory_steps == 0:
                self._train_network()

            self.last_action = [action_id, args, arg_types]

        return FunctionCall(action_id, args)


    def _sample_action(self,
                       screen_features,
                       minimap_features,
                       flat_features,
                       available_actions):
        '''
        Sample action and args from policy
        Returns --
        action_id: index of action
        args: for spatial x,y coords; for non-spatial output from self.network.argument_policy
        arg_types: actions.FUNCTION_TYPES[FUNCTIONS[action_id].function_type] where action id is an index
        '''
        screen_features = np.expand_dims(screen_features, 0)
        minimap_features = np.expand_dims(minimap_features, 0)
        flat_features = np.expand_dims(flat_features, 0)

        action_mask = np.zeros(len(FUNCTIONS), dtype=np.int32)
        action_mask[available_actions] = 1

        feed_dict = {
        self.network.screen_inputs: screen_features,
        self.network.minimap_inputs: minimap_features,
        self.network.flat_inputs: flat_features}

        function_id_policy = self.sess.run(self.network.policy, feed_dict=feed_dict)
        function_id_policy *= action_mask
        function_ids = np.arange(len(function_id_policy))

        # renormalize distribution over function identifiers
        function_id_policy /= np.sum(function_id_policy)

        # sample function identifier
        action_id = np.random.choice(function_ids, p=np.squeeze(function_id_policy))  # why do i need to squeeze here?

        # sample function arguments:
        arg_types = FUNCTION_TYPES[FUNCTIONS[action_id].function_type]
        args = []
        for arg_type in arg_types:
            if len(arg_type.sizes) > 1:
                # this is a spatial action
                # i think this stuff looks good, but
                x_policy = self.sess.run(self.network.argument_policy[str(arg_type) + 'x'],feed_dict=feed_dict)
                y_policy = self.sess.run(self.network.argument_policy[str(arg_type) + 'y'], feed_dict=feed_dict)

                x_policy = np.squeeze(x_policy)  # WHY SQUEEZE? BATCHSIZE?  squeeze bc it comes in as 2d (e.g: [[...]])
                x_ids = np.arange(len(x_policy))
                x = np.random.choice(x_ids, p=x_policy)

                y_policy = np.squeeze(y_policy)
                y_ids = np.arange(len(y_policy))
                y = np.random.choice(y_ids, p=y_policy)
                args.append([x,y])

            else:  # would like to understand this a bit better
                arg_policy = self.sess.run(self.network.argument_policy[str(arg_type)], feed_dict=feed_dict)
                arg_policy = np.squeeze(arg_policy)
                arg_ids = np.arange(len(arg_policy))
                arg_idx = np.random.choice(arg_ids, p=arg_policy)
                args.append([arg_idx])

        return action_id, args, arg_types


    def _handle_episode_end(self):
        ''' save weights and write summaries, trains network, and increments episode '''
        feed_dict = self._train_network(terminal=True)

        self.total_steps = self.steps + self.initial_steps
        self.all_rewards.append(self.reward)

        self.network.increment_global_episode_op(self.sess)
        self.network.save_model(self.sess)
        self.network.write_summary(self.sess, self.global_episode, self.reward, feed_dict)  # ***

        with open(self.save_file, 'wb') as f:
             pickle.dump(self.total_steps, f)
        if self.episodes % 20 == 0:
            self.logger.info(f'Max reward last 20 episodes: {np.max(self.all_rewards[-20:])}')
            self.logger.info(f'Mean reward last 20 episodes: {np.mean(self.all_rewards[-20:])}')

        self.logger.info('Model saved and summary written')


    def _train_network(self, terminal=False):
        ''' trains network with feed_dict from _get_batch '''
        feed_dict = self._get_batch(terminal) 
        self.network.optimizer_op(self.sess, feed_dict)
        return feed_dict 


    def _get_batch(self, terminal=False):
        ''' returns dict of ... '''
        # state
        screen = [state_[0] for state_ in self.state_buffer]
        minimap = [state_[1] for state_ in self.state_buffer]
        flat = [state_[2] for state_ in self.state_buffer]

        # actions and args. remember, last_action = [action, args, arg_types]
        actions = [act_arg[0] for act_arg in self.action_buffer]  # action_id, args, arg_types
        args = [act_arg[1] for act_arg in self.action_buffer]
        arg_types = [act_arg[2] for act_arg in self.action_buffer]

        actions = np.eye(len(FUNCTIONS))[actions]  # one-hot actions 

        # rewards
        raw_rewards = list(self.reward_buffer)
        if terminal:
            value = 0
        else:
            value = np.squeeze(self.sess.run(
                self.network.value_estimate,
                feed_dict={self.network.screen_inputs: screen[-1:],  # why [-1:] -- i think same as reversing rewards
                           self.network.minimap_inputs: minimap[-1:],
                           self.network.flat_inputs: flat[-1:]}))

        discounted_rewards = []
        for reward in list(raw_rewards)[::-1]:
            value = reward + self.gamma * value  # current + reward to go
            discounted_rewards.append(value)

        feed_dict = {self.network.screen_inputs: screen,
                     self.network.minimap_inputs: minimap,
                     self.network.flat_inputs: flat,
                     self.network.actions: actions,
                     self.network.reward: discounted_rewards}

        # add args and arg_types to feed_dict
        network_args = self.network.arguments
        batch_size = len(arg_types)  # will always be 1 per action, even for no_ops

        for arg_type in sc2_actions.TYPES:  # first populate feed_dict with 0s
            if len(arg_type.sizes) > 1:
                if arg_type in SCREEN_TYPES:
                    x_size = SCREEN_DIMS[0]  #  screen pixels (x)
                    y_size = SCREEN_DIMS[1]
                elif arg_type in MINIMAP_TYPES:
                    x_size = MINIMAP_DIMS[0]
                    y_size = MINIMAP_DIMS[1] # minimap pixles (y)

                feed_dict[network_args[str(arg_type) + "x"]] = np.zeros(
                    (batch_size, x_size))
                feed_dict[network_args[str(arg_type) + "y"]] = np.zeros(
                    (batch_size, y_size))
            else:
                feed_dict[network_args[str(arg_type)]] = np.zeros(
                    (batch_size, arg_type.sizes[0]))

        for step in range(batch_size):  # then one_hot encode args
            for i, arg_type in enumerate(arg_types[step]):
                if len(arg_type.sizes) > 1:
                    arg_key_x = network_args[str(arg_type) + "x"]
                    feed_dict[arg_key_x][step, args[step][i][0]] = 1 

                    arg_key_y = network_args[str(arg_type) + "x"]
                    feed_dict[arg_key_y][step, args[step][i][1]] = 1
                else:
                    arg_key = network_args[str(arg_type)]
                    feed_dict[arg_key][step, args[step][i][0]] = 1

        return feed_dict  


    def _init_op(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)


    def _init_logger(self, dir, dir2=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(dir, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if dir2:
            self.logger2 = logging.getLogger('second_logger')
            self.logger2.setLevel(logging.INFO)
            file_handler = logging.FileHandler(dir2, mode='w')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger2.addHandler(file_handler)
