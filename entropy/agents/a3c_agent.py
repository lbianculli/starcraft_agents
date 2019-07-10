import os
import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import utils as U
import networks_2 as networks

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents import base_agent


SCREEN_FEATURES = features.SCREEN_FEATURES
MINIMAP_FEATURES = features.MINIMAP_FEATURES
SCREEN_TYPES = [actions.TYPES[0], actions.TYPES[2]]
MINIMAP_TYPES = [actions.TYPES[1]]

class A3CAgent(base_agent.BaseAgent):
  """An agent specifically for solving the mini-game maps."""
  def __init__(self,
               training=True,
               msize=32,
               ssize=32,
               value_regularisation=0.5,
               entropy_regularisation=0.01,
               logdir="./log/info_logs.log",
               name='A3C/A3CAgent'):
    super(A3CAgent, self).__init__()
    # tf.reset_default_graph()
    self.training = training
    self.summary = []
    self.name = name

    # Minimap size, screen size and info size
    assert msize == ssize
    self.msize = msize
    self.ssize = ssize
    self.value_regularisation = value_regularisation
    self.entropy_regularisation = entropy_regularisation
    self.isize = len(actions.FUNCTIONS)
    if logdir:
      self._init_logger(logdir)


  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
      init_op = tf.global_variables_initializer()
      self.sess.run(init_op)


  def build(self, reuse, dev):  # chaging this around for now. need a2c to work first
    with tf.variable_scope(self.name) and tf.device(dev):  # A3C/A3CAgent/var_name
      if reuse:
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope().reuse

      # Set inputs of networks
      self.score = tf.placeholder(tf.int32, [], name='score')
      self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      net = networks.build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS))
      # will below give me nonetype error too? if so it is b/c arguments and/or argument_policy 100%
      self.spatial_policy, self.non_spatial_policy, self.state_representation, self.value = net

      # Set targets and masks
      self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, self.isize], name='valid_non_spatial_action')  # these match w. actions in previous
      self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, self.isize], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

      # Compute log probability -- what do these look like exactly?
      spatial_action_prob = tf.reduce_sum(self.spatial_policy * self.spatial_action_selected, axis=1)
      spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
      non_spatial_action_prob = tf.reduce_sum(self.non_spatial_policy * self.non_spatial_action_selected, axis=1)
      valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_policy * self.valid_non_spatial_action, axis=1)
      valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
      non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
      non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
      self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
      self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))
      self.summary.append(tf.summary.histogram('spatial_action_log_prob', spatial_action_log_prob))
      self.summary.append(tf.summary.histogram('non_spatial_action_log_prob', non_spatial_action_log_prob))
      # self.logger.info(f"non_spatial_action_log_prob: {non_spatial_action_log_prob}")
      # self.logger.info(f"spatial_action_selected: {self.spatial_action_selected}")  # how does spatial_action_selected look, probs?

      # Compute losses, more details in https://arxiv.org/abs/1602.01783
      # Policy loss and value loss
      action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
      advantage = tf.stop_gradient(self.value_target - self.value)
      policy_loss = - tf.reduce_mean(action_log_prob * advantage)
      value_loss = - tf.reduce_mean(self.value * advantage)
      entropy = tf.reduce_sum(self.non_spatial_policy * tf.log(self.non_spatial_policy), name='entropy')
      self.summary.append(tf.summary.scalar("Entropy", entropy))

      self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
      self.summary.append(tf.summary.scalar('value_loss', value_loss))

      # TODO: policy penalty/entropy
      # loss = policy_loss + value_loss
      loss = tf.add_n([
        policy_loss, self.value_regularisation * value_loss, self.entropy_regularisation * entropy],
        name="loss")
      self.summary.append(tf.summary.scalar("total_loss", loss))
      # Build the optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      grads = opt.compute_gradients(loss)
      cliped_grad = []
      for grad, var in grads:
        self.summary.append(tf.summary.histogram(var.op.name, var))
        self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
        grad = tf.clip_by_norm(grad, 10.0)  # is this an ideal value to clip with?
        cliped_grad.append([grad, var])
      self.train_op = opt.apply_gradients(cliped_grad)
      self.summary_op = tf.summary.merge(self.summary)

      self.saver = tf.train.Saver(max_to_keep=10)

      self.argument_policy = dict()  
      self.arguments = dict()
      for arg_type in actions.TYPES:
        # for spatial actions, represent each dimension independently
        # what if instead of making the output units, i make it a smaller number
        # then do something similar to what eps greedy is doing now?
        if len(arg_type.sizes) > 1:  # if spatial
          if arg_type in SCREEN_TYPES:
            units = self.ssize
          elif arg_type in MINIMAP_TYPES:
            units = self.msize

          arg_policy_x = layers.fully_connected(
            self.state_representation,
            num_outputs=units,
            activation_fn=tf.nn.softmax)

          arg_policy_y = layers.fully_connected(
              self.state_representation,
              num_outputs=units,
              activation_fn=tf.nn.softmax)

          self.argument_policy[str(arg_type) + "x"] = arg_policy_x
          self.argument_policy[str(arg_type) + "y"] = arg_policy_y

          arg_placeholder_x = tf.placeholder(
              tf.float32,
              shape=[None, units])

          arg_placeholder_y = tf.placeholder(
              tf.float32,
              shape=[None, units])

          self.arguments[str(arg_type) + "x"] = arg_placeholder_x
          self.arguments[str(arg_type) + "y"] = arg_placeholder_y

        else:  # if non spatial
          arg_policy = layers.fully_connected(self.state_representation,
              num_outputs=arg_type.sizes[0],
              activation_fn=tf.nn.softmax)

          self.argument_policy[str(arg_type)] = arg_policy

          arg_placeholder = tf.placeholder(
              tf.float32,
              shape=[None, arg_type.sizes[0]])

          self.arguments[str(arg_type)] = arg_placeholder


  def reset(self):
    # Epsilon schedule. first for functions, second for coords
    self.epsilon = [0.05, 0.2]


  def step(self, obs):  # action selection is in here
    minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
    minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
    screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
    screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    # TODO: only use available actions
    info = np.zeros([1, self.isize], dtype=np.float32)
    info[0, obs.observation['available_actions']] = 1

    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    
    # Select an action and a spatial target.
    valid_actions = np.zeros(self.isize, dtype=np.int32)
    valid_actions[obs.observation['available_actions']] = 1
    function_id_policy, spatial_policy = self.sess.run(
      [self.non_spatial_policy, self.spatial_policy],
      feed_dict=feed)
    
    self.logger.info(f"spatial_policy unraveled: {spatial_policy}")
    function_id_policy = function_id_policy.ravel()  # .ravel flattens the input into 1D array
    spatial_policy = spatial_policy.ravel()
    self.logger.info(f"spatial_policy .raveled: {spatial_policy}")  # this will help with target below
    function_id_policy *= valid_actions

    function_ids = np.arange(len(function_id_policy))
    function_id_policy /= np.sum(function_id_policy) 
#     act_id = valid_actions[np.argmax(non_spatial_policy[valid_actions])]
    act_id = np.random.choice(function_ids, p=np.squeeze(function_id_policy))
    target = np.argmax(spatial_policy) # ***
    target = [int(target // self.ssize), int(target % self.ssize)]  # not sure

    if False:  
      self.logger.info(f"if false: {actions.FUNCTIONS[act_id].name, target}")

    # Epsilon greedy exploration. Keeping this to see if it works
    # basically, if eps greedy: take the target and move it left/right and up/down 4 px
    if self.training and np.random.rand() < self.epsilon[1]:
      dy = np.random.randint(-4, 5)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy))) # make sure target is within possible pxl range
      dx = np.random.randint(-4, 5)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    args = []
    # args: A list of the types of args passed to function_type
    for arg in actions.FUNCTIONS[act_id].args:
      if arg.name in ('screen', 'minimap', 'screen2'):
          # x_policy = self.sess.run(
          #     self.argument_policy[str(arg) + "x"],
          #     feed_dict=feed)

          # y_policy = self.sess.run(
          #     self.argument_policy[str(arg) + "y"],
          #     feed_dict=feed)

          # x_policy = np.squeeze(x_policy)
          # x_ids = np.arange(len(x_policy))
          # x = np.random.choice(x_ids, p=x_policy)

          # y_policy = np.squeeze(y_policy)
          # y_ids = np.arange(len(y_policy))
          # y = np.random.choice(y_ids, p=y_policy)
          # args.append([x, y])
          args.append([target[1], target[0]])
          self.logger.info(f"target coords: {[target[1], target[0]]}")
      else:
          arg_policy = self.sess.run(
              self.argument_policy[str(arg)],
              feed_dict=feed)
          arg_policy = np.squeeze(arg_policy)
          arg_ids = np.arange(len(arg_policy))
          arg_index = np.random.choice(arg_ids, p=arg_policy)
          args.append([arg_index])  
          self.logger.info(f"arg: index: {arg_index}") 
#           args.append([0])
    
    # sizes: The max+1 of each of the dimensions this argument takes.
    return actions.FunctionCall(act_id, args)  #  args should be int from (0, arg.size)


  def update(self, replay_buffer, discount, lr, counter):
    """ replay_buffer is list of recorders, which are lists of (s, a, s`) """
    # Compute R, which is value of the last observation
    obs = replay_buffer[-1][-1]  # last state of most recent loop
    if obs.last():
      R = 0
    else:
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value, feed_dict=feed)[0]

    # Compute targets and masks
    minimaps = []
    screens = []
    infos = []

    value_target = np.zeros([len(replay_buffer)], dtype=np.float32)
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(replay_buffer)], dtype=np.float32)
    spatial_action_selected = np.zeros([len(replay_buffer), self.ssize**2], dtype=np.float32)
    valid_non_spatial_action = np.zeros([len(replay_buffer), self.isize], dtype=np.float32)
    non_spatial_action_selected = np.zeros([len(replay_buffer), self.isize], dtype=np.float32)

    replay_buffer.reverse()
    for i, [obs, action, next_obs] in enumerate(replay_buffer):
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      self.reward = obs.reward
      act_id = action.function
      act_args = action.arguments

      value_target[i] = self.reward + discount * value_target[i-1]

      valid_actions = obs.observation["available_actions"]
      valid_non_spatial_action[i, valid_actions] = 1
      non_spatial_action_selected[i, act_id] = 1

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):  # set up masks
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1

    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)
    
    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr,} 
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
    self.summary_writer.add_summary(summary, counter)


  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)  # note: episode count


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])


  def _init_logger(self, dir):
      self.logger = logging.getLogger(__name__)
      self.logger.setLevel(logging.INFO)
      file_handler = logging.FileHandler(dir, mode='w')
      file_handler.setLevel(logging.INFO)
      formatter = logging.Formatter('%(levelname)s - %(message)s')
      file_handler.setFormatter(formatter)
      self.logger.addHandler(file_handler)

