import numpy as np
import os
import tensorflow as tf
from collections import deque
import logging

import dqn_1_1 as nets

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from pysc2.lib import features


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


class DQNMoveOnlyAgent(base_agent.BaseAgent):
    def __init__(self,
                 learning_rate=1e-4,
                 value_gradient_strength=.1,  # check these
                 regularization_strength=.05,
                 gamma=.99,
                 epsilon_decay_steps=1000,
                 epsilon_max = 1.0,
                 epsilon_min = 0.01,
                 training_start=500,
                 training_steps=50,  # steps between training ops. whats good value here? kept low so that cpu could run
                 training=True,
                 prioritized=False,
                 prioritized_alpha = .6,
                 prioritized_beta = .4,
                 save_dir='/home/lbianculli/sc_bot/minigames/logs/ckpts/',
                 save_file='/home/lbianculli/sc_bot/minigames/collect_minerals/logs/network_saves',
                 ckpt_name='collect_minerals_6-25',
                 summary_path='/home/lbianculli/sc_bot/minigames/logs/',
                 logdir='/home/lbianculli/sc_bot/minigames/logs/variable_logs/',
                 logdir2='/home/lbianculli/sc_bot/minigames/logs/variable_logs2/',
                 buffer_path='/home/lbianculli/sc_bot/minigames/collect_minerals/logs/buffers/buffer_6-25',
                 log=True):
        super(DQNMoveOnlyAgent, self).__init__()  # what does this do again?

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
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.training_start = training_start
        self.training_steps = training_steps
        self.training = training
        self.save_file = save_file
        self.prioritized = prioritized
        self.buffer_path = buffer_path

        if log:
            self._init_logger(logdir, logdir2)

        self.save_path = save_dir + ckpt_name + '.ckpt'

        if self.prioritized:
            self.buffer_file = buffer_path + '_prioritized.p'
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters, initial_p=self.prioritized_beta, final_p=1.0)
        else:
            self.buffer_file = buffer_path + '.p'

        # load and set epsilon
        if os.path.isfile(self.save_file + '.npy'):
            self.epsilon, self.initial_step = np.load(self.save_file + '.npy')  # can i just use loaded step for epsilon as well?
            print(f'epsilon loaded: {self.epsilon}')
        else:
            self.epsilon = 1.0
            self.initial_step = 0
        self.epsilons = [self.epsilon]

        tf.reset_default_graph()
        self.online_network = nets.AlphaCNN(
            screen_dims=FEATURE_SCREEN_SIZE,
            minimap_dims=FEATURE_MINIMAP_SIZE,
            learning_rate=1e-4,
            value_gradient_strength=value_gradient_strength,
            regularization_strength=regularization_strength,
            save_path=self.save_path,
            summary_path=summary_path
            )
        self.logger.info('Network initialization complete.')

        self.sess = tf.Session()
        if os.path.isfile(self.save_path + '.index'):
            self.online_network.load(self.sess)
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
            self.global_episode = self.online_network.global_episode.eval(self.sess)


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
        action_id, args, arg_types = self._sample_action(obs,
                                                         screen_features,
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


    def _epsilon_greedy_action(self, state, available_actions, epsilon=1.0):
        ### how to coordinate w/in if so i can use cleanly above? ###
        ''' Choose an action from the state with eps greedy '''
        self.total_steps = int(self.steps + self.initial_step)  # agent step differs from game step due to step mul.
        fraction = min(float(self.total_steps) / self.epsilon_decay_steps, 1.0)
        new_epsilon = epsilon + fraction * (self.epsilon_min - self.epsilon_max)
        self.epsilons.append(new_epsilon)  # for logging/loading

        if new_epsilon > np.random.rand():  # chooses random action
            action_id = np.random.choice(available_actions)
            # x = np.random.randint(0, FEATURE_SCREEN_SIZE[0])
            # y = np.random.randint(0, FEATURE_SCREEN_SIZE[1])

            return action_id

        else:  # smthn like this. not important until the rest works anyway
            inputs = np.expand_dims(state, 0)  # state = obs (from above)  (???)  # below: flat is output in old code
            q_values = self.sess.run(self.online_network.flat_inputs, feed_dict={self.online_network.inputs:inputs})  # flatten for unravel
            action_id = tf.argmax(q_values)  # best bet might be changing this? has to be an easier way
            # x, y = np.unravel_index(best_action, FEATURE_SCREEN_SIZE)  # not entirely sure why this is best, complicated

            return action_id


    def _sample_action(self,
                       obs,
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
        self.online_network.screen_inputs: screen_features,
        self.online_network.minimap_inputs: minimap_features,
        self.online_network.flat_inputs: flat_features}

        ### ---/ From here on out is the tricky part /--- ###
        # definitely need to use something similar to below to account for random actions

        network_pred_q = self.sess.run(self.online_network.pred_q, feed_dict=feed_dict)
        network_pred_q *= action_mask
        function_ids = np.arange(len(network_pred_q))

        # renormalize distribution over function identifiers
        # function_id_policy /= np.sum(function_id_policy)

        # sample function identifier  -- below will pick random. need to incorporate eps greedy as well
        action_id = self._epsilon_greedy_action(obs, available_actions)  # not even sure about other args
        # action_id = np.random.choice(function_ids, p=np.squeeze(function_id_policy))  # why do i need to squeeze here?

        # sample function arguments:
        arg_types = FUNCTION_TYPES[FUNCTIONS[action_id].function_type]
        args = []
        for arg_type in arg_types:
            if len(arg_type.sizes) > 1:
                self.logger.info('SPATIAL')
                # this is a spatial action
                # self.logger.info(f'ARGUMENT POLICY DICT: {self.network.argument_policy.keys()}')
                # i think this stuff looks good, but
                x_policy = self.sess.run(self.online_network.argument_policy[str(arg_type) + 'x'],feed_dict=feed_dict)
                y_policy = self.sess.run(self.online_network.argument_policy[str(arg_type) + 'y'], feed_dict=feed_dict)

                x_policy = np.squeeze(x_policy)  # WHY SQUEEZE? BATCHSIZE?  squeeze bc it comes in as 2d (e.g: [[...]])
                x_ids = np.arange(len(x_policy))
                x_id = np.random.choice(x_ids, p=x_policy)

                y_policy = np.squeeze(y_policy)
                y_ids = np.arange(len(y_policy))
                y_id = np.random.choice(y_ids, p=y_policy)
                args.append([x_id,y_id])

            else:
                self.logger.info('NON-SPATIAL')
                arg_policy = self.sess.run(self.online_network.argument_policy[str(arg_type)], feed_dict=feed_dict)
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
            value = 0 # dont think i still need
            done_mask = 1
        else:
            value = np.squeeze(self.sess.run(
                self.online_network.pred_q,
                feed_dict={self.online_network.screen_inputs: screen[-1:],  # why [-1:]?? whats the  difference?
                           self.online_network.minimap_inputs: minimap[-1:],
                           self.online_network.flat_inputs: flat[-1:]}))

        # self.target_q_action = self.reward + (1-self.done_mask) * self.gamma *tf.reduce_max(target_q, axis=1)  # ***

        upd_rewards = []
        # n-step discounted rewards from 1 < n < trajectory_training_steps. *** feel like this isnt right, needs exp(i)?
        for i, reward in enumerate(raw_rewards):
            value = reward + self.gamma * value  # this differs from above
            upd_rewards.append(value)

        feed_dict = {self.online_network.screen_inputs: screen,
                     self.online_network.minimap_inputs: minimap,
                     self.online_network.flat_inputs: flat,
                     self.online_network.actions: actions,
                     self.online_network.reward: upd_rewards}

        # add args and arg_types to feed_dict
        net_args = self.online_network.arguments  #  dict of phs of shape [None, px], keyed by arg_type  (check)
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

        self.logger2.info(f'Net args: {net_args}\n')  # matches above - [None, px] ph
        self.logger2.info(f'Feed dict 1 keys length: {len(feed_dict.keys())}\n')
        self.logger2.info(f'Feed dict 1: {feed_dict}\n')  # [None, 17, 32, 32] array -- full_game_CNN/screen_inputs

        # then one_hot encode args
        for step in range(batch_size):
            for i, arg_type in enumerate(arg_types[step]):
                if len(arg_type.sizes) > 1:
                    arg_key_x = net_args[str(arg_type) + "x"]  # get the VALUE of net_args for this arg
                    feed_dict[arg_key_x][step, args[step][i][0]] = 1  # haha this is rough... why step? temporal dimension?

                    arg_key_y = net_args[str(arg_type) + "x"]
                    feed_dict[arg_key_y][step, args[step][i][1]] = 1
                else:
                    arg_key = net_args[str(arg_type)]
                    feed_dict[arg_key][step, args[step][i][0]] = 1
        self.logger.info(f'Feed dict final keys length: {len(feed_dict.keys())}\n')  #
        self.logger.info(f'Feed dict final: {feed_dict}\n')

        return feed_dict  # ending feed dict will have phs with 1 at intersection of coord and action idx


    def _train_network(self, terminal=False):
        ''' trains network with feed_dict from _get_batch '''
        feed_dict = self._get_batch(terminal)  # terminal if episode end
        self.online_network.optimizer_op(self.sess, feed_dict)
        # dont see why i cant just replace a2c code in a2c_net with dqn code...
        # if i am understanding this correctly

        return feed_dict


    def _handle_episode_end(self):
        ''' save weights and write summaries '''

        # train network and increment episode
        feed_dict = self._train_network(terminal=True)  # should i be using this for something? ****
        self.online_network.increment_global_episode_op(self.sess)
        self.online_network.save_model(self.sess)
        self.online_network.write_summary(self.sess, feed_dict)
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

