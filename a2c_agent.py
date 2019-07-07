import tensorflow as tf
from collections import deque
import logging
import os

import a2c_net as nets
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

print(tf.__version__)

# agent interface settings (defaults specified in pysc2.bin.agent)
FEATURE_SCREEN_SIZE = [32, 32]
FEATURE_MINIMAP_SIZE = [32, 32]

# pysc2 convenience
FUNCTIONS = sc2_actions.FUNCTIONS
FUNCTION_TYPES = sc2_actions.FUNCTION_TYPES
FunctionCall = sc2_actions.FunctionCall

# manually state the argument types which take points on screen/minimap
SCREEN_TYPES = [sc2_actions.TYPES[0], sc2_actions.TYPES[2]]
MINIMAP_TYPES = [sc2_actions.TYPES[1]]


class a2cAgent(base_agent.BaseAgent):
    def __init__(self,
                 learning_rate=1e-4,
                 value_gradient_strength=.1,  # check these
                 regularization_strength=.05,
                 gamma=.99,
                 training_start=500,
                 training_steps=250,  # steps between training ops. whats good value here? kept low so that cpu could run
                 training=True,
                 save_dir='C:/Users/lbianculli/venv1/sc_bot/full_games/logs/ckpts/',
                 ckpt_name='collect_minerals_5-09',
                 summary_path='C:/Users/lbianculli/venv1/sc_bot/full_games/logs/',
                 logdir='C:/Users/lbianculli/venv1/sc_bot/full_games/logs/variable_logs.txt/',
                 logdir2='C:/Users/lbianculli/venv1/sc_bot/full_games/logs/variable_logs2.txt/',
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
        self.training_start = training_start
        self.training_steps = training_steps
        self.training = training

        if log:
            self._init_logger(logdir, logdir2)

        self.save_path = save_dir + ckpt_name + '.ckpt'

        tf.reset_default_graph()
        self.network = nets.AlphaCNN(
            screen_dims=FEATURE_SCREEN_SIZE,
            minimap_dims=FEATURE_MINIMAP_SIZE,
            learning_rate=1e-4,
            value_gradient_strength=value_gradient_strength,
            regularization_strength=regularization_strength,
            save_path=self.save_path,
            summary_path=summary_path
            )
        self.logger.info('Network initialization complete.')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if os.path.isfile(self.save_path + '.index'):
            self.network.load(self.sess)
        else:
            self._init_op()


    def reset(self):
        # reset isnt currently running.
        self.episodes += 1
        self.steps = 0
        self.reward = 0

        if self.training:
            self.last_action = None
            self.state_buffer = deque(maxlen=self.training_steps)
            self.action_buffer = deque(maxlen=self.training_steps)
            self.reward_buffer = deque(maxlen=self.training_steps)
            self.global_episode = self.network.global_episode.eval(self.sess)


    def step(self, obs):
        self.steps += 1
        if self.steps % 100 == 0:
            self.logger.info(f'STEP NUMBER: {self.steps}')
        self.reward += obs.reward

        if self.training and obs.last():
            self._handle_episode_end()

        observation = obs.observation
        screen_features = observation.feature_screen
        minimap_features = observation.feature_minimap
        flat_features = observation.player  # makes sense
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
            if (self.steps % self.training_steps == 0) and (self.steps > self.training_start):
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

        function_id_policy = self.sess.run(self.network.function_policy, feed_dict=feed_dict)
        function_id_policy *= action_mask
        function_ids = np.arange(len(function_id_policy))

        # renormalize distribution over function identifiers
        function_id_policy /= np.sum(function_id_policy)
        # self.logger.info(f'FUNCTION_ID_POLICY: {function_id_policy}')

        # sample function identifier
        action_id = np.random.choice(function_ids, p=np.squeeze(function_id_policy))  # why do i need to squeeze here?

        # sample function arguments:
        arg_types = FUNCTION_TYPES[FUNCTIONS[action_id].function_type]
        args = []
        for arg_type in arg_types:
            if len(arg_type.sizes) > 1:
                self.logger.info('SPATIAL')
                # this is a spatial action
                # self.logger.info(f'ARGUMENT POLICY DICT: {self.network.argument_policy.keys()}')
                # i think this stuff looks good, but
                x_policy = self.sess.run(self.network.argument_policy[str(arg_type) + 'x'],feed_dict=feed_dict)
                y_policy = self.sess.run(self.network.argument_policy[str(arg_type) + 'y'], feed_dict=feed_dict)

                x_policy = np.squeeze(x_policy)  # WHY SQUEEZE? BATCHSIZE?  squeeze bc it comes in as 2d (e.g: [[...]])
                x_ids = np.arange(len(x_policy))
                x_id = np.random.choice(x_ids, p=x_policy)

                y_policy = np.squeeze(y_policy)
                y_ids = np.arange(len(y_policy))
                y_id = np.random.choice(y_ids, p=y_policy)
                args.append([x_id,y_id])

            else:
                self.logger.info('NON-SPATIAL')
                arg_policy = self.sess.run(self.network.argument_policy[str(arg_type)], feed_dict=feed_dict)
                arg_policy = np.squeeze(arg_policy)
                arg_ids = np.arange(len(arg_policy))
                arg_id = np.random.choice(arg_ids, p=arg_policy)
                args.append([arg_id])

        self.logger.info(f'Action_ids: {action_id}')
        self.logger.info(f'args: {args}')
        self.logger.info(f'arg_types: {arg_types}\n')


        return action_id, args, arg_types


    def _get_batch(self, terminal=False):
        ''' returns dict of ... '''
        # state
        screen = [state_[0] for state_ in self.state_buffer]
        minimap = [state_[1] for state_ in self.state_buffer]
        flat = [state_[2] for state_ in self.state_buffer]

        # actions and args. remember, last_action = [action, args, arg_types]
        actions = [act_arg[0] for act_arg in self.action_buffer]  # action_id, args, arg_types
        actions = np.eye(len(FUNCTIONS))[actions]  # puts a 1 at index specified by action_id value

        args = [act_arg[1] for act_arg in self.action_buffer]
        arg_types = [act_arg[2] for act_arg in self.action_buffer]
        # self.logger.info(f'ARGS: {args}')  # look like coords
        # self.logger.info(f'ARG TYPES: {arg_types}')  # these are arg types from before. include all function argument types

        # rewards
        raw_rewards = list(self.reward_buffer)
        if terminal:
            value = 0
        else:
            value = np.squeeze(self.sess.run(
                self.network.value_estimate,
                feed_dict={self.network.screen_inputs: screen[-1:],  # why [-1:]?? whats the  difference?
                           self.network.minimap_inputs: minimap[-1:],
                           self.network.flat_inputs: flat[-1:]}))

        upd_rewards = []
        # n-step discounted rewards from 1 < n < trajectory_training_steps. *** feel like this isnt right, needs exp(i)?
        for i, reward in enumerate(raw_rewards):
            value = reward + self.gamma * value
            upd_rewards.append(value)

        feed_dict = {self.network.screen_inputs: screen,
                     self.network.minimap_inputs: minimap,
                     self.network.flat_inputs: flat,
                     self.network.actions: actions,
                     self.network.reward: upd_rewards}

        # add args and arg_types to feed_dict
        net_args = self.network.arguments  #  dict of phs of shape [None, px], keyed by arg_type
        batch_size = len(arg_types)  # will always be 1 per action, even for no_ops

        for arg_type in sc2_actions.TYPES:
            if len(arg_type.sizes) > 1:
                if arg_type in SCREEN_TYPES:
                    x_size = FEATURE_SCREEN_SIZE[0]  #  screen pixels (x)
                    y_size = FEATURE_SCREEN_SIZE[1]
                elif arg_type in MINIMAP_TYPES:
                    x_size = FEATURE_MINIMAP_SIZE[0]
                    y_size = FEATURE_MINIMAP_SIZE[1] # minimap pixles (y)

                feed_dict[net_args[str(arg_type) + "x"]] = np.zeros(
                    (batch_size, x_size))
                feed_dict[net_args[str(arg_type) + "y"]] = np.zeros(
                    (batch_size, y_size))

            else:
                feed_dict[net_args[str(arg_type)]] = np.zeros(
                    (batch_size, arg_type.sizes[0]))

        self.logger2.info(f'Net args: {net_args}\n')
        self.logger2.info(f'Feed dict 1: {feed_dict}\n')

        # then one_hot encode args
        for step in range(batch_size):
            for i, arg_type in enumerate(arg_types[step]):
                if len(arg_type.sizes) > 1:
                    arg_key_x = net_args[str(arg_type) + "x"]
                    feed_dict[arg_key_x][step, args[step][i][0]] = 1

                    arg_key_y = net_args[str(arg_type) + "x"]
                    feed_dict[arg_key_y][step, args[step][i][1]] = 1
                else:
                    arg_key = net_args[str(arg_type)]
                    feed_dict[arg_key][step, args[step][i][0]] = 1
        # self.logger.info(f'FEED DICT KEYS: {feed_dict.keys()}')  # these are just placeholders
        self.logger.info(f'Feed dict final: {feed_dict}\n')
        return feed_dict  # ending feed dict will have phs with corresponding binary (?)



    def _train_network(self, terminal=False):
        ''' trains network with feed_dict from _get_batch '''
        feed_dict = self._get_batch(terminal)  # terminal if episode end
        self.network.optimizer_op(self.sess, feed_dict)

        return feed_dict  # use this in handle_episode_end


    def _handle_episode_end(self):
        ''' save weights and write summaries '''

        # train network and increment episode
        feed_dict = self._train_network(terminal=True)
        self.network.increment_global_episode_op(self.sess)
        self.network.save_model(self.sess)
        self.network.write_summary(self.sess, feed_dict)
        self.logger.info('Model saved and summary written')


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
