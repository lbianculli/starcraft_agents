"""Deep Q-learning agents."""
import numpy as np
import os
import tensorflow as tf
import logging
from absl import flags
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import pickle
import time
from checkpoint_tools import print_tensors_in_checkpoint_file
from baselines.common.schedules import LinearSchedule

from dqn_utils import LinearSchedule
from dqn_value_estimator import PlayerRelativeMovementCNN


FEATURE_SCREEN_SIZE = [32,32]  # should this be [32, 32] or 32? ??? clone his tomorrow to see if it works
FEATURE_MINIMAP_SIZE = 32  # not actually needed for minigames
FUNCTIONS = sc2_actions.FUNCTIONS


class DQNMoveOnlyAgent(base_agent.BaseAgent):
    ''' DQN that takes in player_relative features and returns movements '''

    def __init__(self,
                 learning_rate,  # could use linearschedule here as well?
                 gamma,
                 epsilon_max,
                 epsilon_min,
                 epsilon_decay_steps=20000,
                 learning_starts=1000,
                 train_freq=4,
                 target_update_freq=5000,
                 max_buffer_size=25000,
                 batch_size=16,
                 prioritized_replay_beta_iters = 10000,  # what is a good value for this?
                 training=True,
                 indicate_nonrandom_action=False,
                 prioritized=True,
                 prioritized_alpha = .7,  #these values are for rank-based prioritization
                 prioritized_beta = .5,
                 epsilon_file='/home/lbianculli/sc_bot/minigames/dqn_minigames/logs/epsilon',
                 save_dir='/home/lbianculli/sc_bot/minigames/dqn_minigames/logs/ckpts/',
                 ckpt_name='collect_minerals_5-09',
                 summary_path='/home/lbianculli/sc_bot/minigames/dqn_minigames/logs/summaries/',
                 buffer_path='/home/lbianculli/sc_bot/minigames/dqn_minigames/logs/buffers/buffer_5-09',
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
        self.epsilon_file = epsilon_file
        self.batch_size = batch_size
        self.log = log
        self.episode_td_errors = []

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
        if os.path.isfile(self.epsilon_file + '.npy'):
            self.epsilon = np.load(self.epsilon_file + '.npy')
            print(f'epsilon loaded: {self.epsilon}')
        else:
            self.epsilon = 1.0
        self.epsilons = [self.epsilon]

        # for saving and loading files
        if save_dir:
            self.online_save_dir = save_dir + 'online/'  # for use in checkpoints
            self.target_save_dir = save_dir + 'target/'

        if ckpt_name:
            self.ckpt_name = ckpt_name

        if summary_path:
            self.online_summary_path = summary_path + 'online/' # for use in TB summaries
            self.target_summary_path = summary_path + 'target/'

        if self.log:
            self.init_logger('/home/lbianculli/sc_bot/minigames/dqn_minigames/logs/variable_logs/')

        # build network
        if save_dir and ckpt_name:
            self.online_save_path = self.online_save_dir + ckpt_name + '.ckpt'
            self.target_save_path = self.target_save_dir + ckpt_name + '.ckpt'
        print("Building models...")
        tf.reset_default_graph()
        self.online_network = PlayerRelativeMovementCNN(spatial_dims=FEATURE_SCREEN_SIZE,
                                                 learning_rate=self.learning_rate,
                                                 save_path=self.online_save_path,
                                                 summary_path=self.online_summary_path,
                                                 name='DQN')
        if self.training:
            # set up target_net and initialize replay buffer
            self.target_network = PlayerRelativeMovementCNN(spatial_dims=FEATURE_SCREEN_SIZE,
                                                            learning_rate=self.learning_rate,
                                                            save_path = self.target_save_path,
                                                            summary_path = self.target_summary_path,
                                                            name='target_network')
        self.sess = tf.Session()
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
                self.replay_buffer = PrioritizedReplayBuffer(max_buffer_size, self.prioritized_alpha)  # what about buffer methods? check baselines examples?
            else:
                self.replay_buffer = ReplayBuffer(max_buffer_size)

        print('online and target models loaded.')
        self._tf_init()

        if self.training:
            self._update_target_network()  # do i still need this?
        else:
            self._tf_init()


    def reset(self):
        ''' reset episode '''
        self.episodes += 1
        self.reward = 0

        if self.training:
            self.last_state = None
            self.last_action = None
            self.episode = self.online_network.global_episode.eval(self.sess)


    def step(self, obs):
        ''' If no units selected, selects army. Otherwise, move. '''
        self.steps += 1
        self.reward = obs.reward
        if self.reward > self.max_reward:
            self.max_reward = self.reward

        # handle terminal steps:
        if self.training and obs.last():
            self._handle_episode_end()

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            state = obs.observation.feature_screen.player_relative

            if self.training:  
                # predict and take an action
                x, y, action = self._epsilon_greedy_action(state, self.epsilon)

                # update online DQN/target network if appropriate
                if (self.steps % self.train_freq == 0) and (len(self.replay_buffer) > self.batch_size):
                    self._train_network()

                if self.agent_step % self.target_update_freq == 0:
                    self._update_target_network()

                # add to replay buffer
                if self.last_state is not None:
                    self.replay_buffer.add(self.last_state, self.last_action, obs.reward, state, 0)

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

        
    def init_logger(self, dir=None):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(dir, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)


    def log_progress(self):
        # ''' log some stats and measurements in a separate file'''
        self.logger.info(f'Game steps completed: {self.steps}')
        self.logger.info(f'Agent steps completed: {self.agent_step}')  # same thing as global step -- + 60 for each ep
        self.logger.info(f'Episodes completed: {self.episode+1}')
        self.logger.info(f'Weight values: {self.batch_weights}')
        self.logger.info(f'Beta value: {self.beta_schedule.value(self.agent_step)}')
        self.logger.info(f'Epsilon: {self.epsilons[-1]}')

        
    def _handle_episode_end(self):
        ''' Save weights, eps, buffer, and write summaries '''
        self.online_network.increment_global_episode_op(self.sess)

        #save online and target nets and write summaries
        if self.episodes % 5 ==  1: 
            self._save_and_clear()
        states, actions, targets = self._get_batch()
        self.online_network.write_summary(self.sess, states, actions, targets, self.reward)

        if self.log:
            self.log_progress()
        print('Model saved and summaries written.')


    def _tf_init(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)


    def _save_and_clear(self):  
        ''' Saves models, epsilon, buffer. cleans directories (NOTIMPLEMENTED'''
        self.online_network.save_model(self.sess)
        self.target_network.save_model(self.sess)

        if self.training:
            np.save(self.epsilon_file + '.npy', self.epsilons[-1])

        with open(self.buffer_file, 'wb') as f:
            pickle.dump(self.replay_buffer, f)  


    def _update_target_network(self):
        ''' get all trainable vars from graph for online and target, zip, then assign from online --> target '''
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DQN')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_network')

        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))

        self.sess.run(update_op)


    def _epsilon_greedy_action(self, state, epsilon=1.0):
        ''' Choose an action from the state with eps greedy '''
        self.agent_step = self.online_network.global_step.eval(self.sess)  # agent step differs from game step due to step mul. Also begins updating after opt
        fraction = min(float(self.agent_step) / self.epsilon_decay_steps, 1.0)
        upd_epsilon = epsilon + fraction * (self.epsilon_min - self.epsilon_max)
        self.epsilons.append(upd_epsilon)  # for logging/loading

        if upd_epsilon > np.random.rand():
            x = np.random.randint(0, FEATURE_SCREEN_SIZE[0])
            y = np.random.randint(0, FEATURE_SCREEN_SIZE[1])

            return x, y, 'random'

        else:
            inputs = np.expand_dims(state, 0)  # state = obs (from above)
            q_values = self.sess.run(self.online_network.flat, feed_dict={self.online_network.inputs:inputs})  # flatten for unravel
            best_action = np.argmax(q_values)
            x, y = np.unravel_index(best_action, FEATURE_SCREEN_SIZE)  # not entirely sure why this is best, complicated

            return x, y, 'nonrandom'


    def _train_network(self):
        # if self.steps >= self.learning_starts:
        states, actions, targets = self._get_batch()
        self.online_network.optimizer_op(self.sess, states, actions, targets)


    def _calc_ranks(self, idxes, td_errors):
        '''
        calcs ranked-based priorities given inputs.
        returns: rank-based priorities and their corresponding indexes as lists
        '''
        #  P(i) = (pi^ a) / sum(pk^a);  p(i) = 1 / rank(i)
        zipped = sorted(list(zip(td_errors, idxes)))
        sorted_idxes = []
        sorted_abs = []
        for pair in zipped:
            sorted_abs.append(pair[0])
            sorted_idxes.append(pair[1])

        ranks = [1/v for v in range(1, len(sorted_abs) + 1)]

        return sorted_idxes, ranks


    def _get_batch(self):
        '''
        uses openai.baselines replay buffer to sample transitions.
        returns batch of observations_t, actions_t and targets (rewards)
        '''
        if self.prioritized:
            # how can i log?
            batch = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(self.agent_step))
            obses, actions, rewards, obses_tp1, dones, self.batch_weights, batch_idxes = batch 

        else:
            batch = self.replay_buffer.sample(self.batch_size)
            obses, actions, rewards, obses_tp1, dones = batch

        actions = np.eye(np.prod(FEATURE_SCREEN_SIZE))[actions] 

        next_output = self.sess.run(self.target_network.output, feed_dict={self.target_network.inputs: obses_tp1})
        targets = [rewards[i] + self.gamma * np.max(next_output[i]) for i in range(self.batch_size)]
        td_errors = self.sess.run([self.online_network.loss2,],
                                feed_dict={self.online_network.inputs: obses,
                            self.online_network.actions: actions,
                            self.online_network.targets: targets})
        td_errors = td_errors[0]
        weighted_errors = self.batch_weights * td_errors
        sorted_idxes, ranks = self._calc_ranks(batch_idxes, weighted_errors)

        self.replay_buffer.update_priorities(sorted_idxes, ranks)

        return obses, actions, targets
