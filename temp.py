import os
import pickle
import json
import logging
import numpy as np
import utils as U
import tensorflow.contrib.layers as layers

from google.protobuf.json_format import MessageToJson
from pysc2.lib import actions


SCREEN_TYPES = [actions.TYPES[0], actions.TYPES[2]]
MINIMAP_TYPES = [actions.TYPES[1]]


class SupervisedParam(object):
  def __init__(self, isize=11, ssize=64, msize=64, parsed_dir='/home/lbianculli/agent_replay_data/'):

    self.parsed_dir = parsed_dir
    self.isize = isize
    self.msize = msize
    self.ssize = ssize
    self.parsed_filenames = os.listdir(self.parsed_dir)  # list of pickled replays
    self.next_index = 0

    self._init_logger(dir="./action_param_log")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)

    self.build_net()
    self.build_opt()

  # every batch corresponding to 1 replay file ***
  def get_batch(self, get_action_id_only=False):
    """
    Loads pickled replay data. Returns screen, minimap, actions, player info and coords for SL step
    """
    full_filename = self.parsed_dir + self.parsed_filenames[self.next_index]  # single replay path

    while os.path.getsize(full_filename) == 0:  # this cant be the best way
      del self.parsed_filenames[self.next_index]
      full_filename = self.parsed_dir+self.parsed_filenames[self.next_index]
      if self.next_index >= len(self.parsed_filenames):
        self.next_index = 0

    self.next_index += 1
    if self.next_index == len(self.parsed_filenames):
      self.next_index = 0

    try:
      replay_data = pickle.load(open(full_filename, "rb"))
    except:
      self.logger.info(f"Skipping replay: {full_filename}")
      return self.get_batch(get_action_id_only)

    loaded_replay_info_json = MessageToJson(replay_data['info'])
    info_dict = json.loads(loaded_replay_info_json)

    # get winner from json
    winner_id = -1
    for info in info_dict['playerInfo']:
      if info['playerResult']['result'] == 'Victory':
        winner_id = int(info['playerResult']['playerId'])
        break

    if winner_id == -1:  # if its a tie
      replay_data = [] # release memory
      return self.get_batch(get_action_id_only)

    minimap_output = []  # set up lists for arrays
    screen_output = []
    action_output = []
    player_info_output = []
    ground_truth_coordinates = []

    for state in replay_data['state']:
      if state['actions'] == []:
        continue

      # player info
      info_temp = np.array(state['player'])
      if info_temp[0] != winner_id:
        continue

      # minimap and screen temps
      m_temp = np.array(state['feature_minimap'], dtype=np.float32)
      m_temp = np.expand_dims(U.preprocess_minimap(m_temp), axis=0)  # will throw error unless parsed at same size
      s_temp = np.array(state['feature_screen'], dtype=np.float32)
      s_temp = np.expand_dims(U.preprocess_screen(s_temp), axis=0)

      # one-hot action_id
      last_action = None
      for action in state['actions']:
        if last_action == action:
          continue

        one_hot = np.zeros((1, 543)) # Not sure where 543 is from. Hardcoding for now
        one_hot[np.arange(1), [action[0]]] = 1

        for param in action[2]:
          if param == [0]:  # seen this before. no_op i think
            continue
          minimap_output.append(m_temp)
          screen_output.append(s_temp)
          action_output.append(one_hot[0])
          player_info_output.append(pi_temp)
          ground_truth_coordinates.append(np.array(param))  # handling would begin here.

    assert(len(minimap_output) == len(ground_truth_coordinates))

    if len(minimap_output) == 0:
      # The replay file only record one person's operation, so if it is
      # the defeated person, we need to skip the replay file
      return self.get_batch(get_action_id_only)

    if get_action_id_only:
      return minimap_output, screen_output, player_info_output, action_output
    else:
      return minimap_output, screen_output, action_output, player_info_output, ground_truth_coordinates


  def build_net(self):
    self.score = tf.placeholder(tf.int32, [], name='score')  # do i need?
    self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')  # 17, 64, 64
    self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
    self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

    self.action_output = tf.placeholder(tf.float32, [None, 543]) # one hot

    # set up network ** still need to change some of these
    screen_filters1 = tf.get_variable(name='sf1', shape=(5, 5, U.screen_channel(), 16))  # hwio
    screen_filters2 = tf.get_variable(name='sf2',shape=(3, 3, 16, 32))
    minimap_filters1 = tf.get_variable(name='mmf1',shape=(5, 5, U.minimap_channel(), 16))
    minimap_filters2 = tf.get_variable(name='mmf2',shape=(3, 3, 16, 32))

    mconv1 = tf.nn.conv2d(tf.transpose(minimap, [0, 2, 3, 1]), minimap_filters1, strides=[1, 1, 1, 1], padding='SAME', name='mconv1')
    mconv2 = tf.nn.conv2d(mconv1, minimap_filters2, strides=[1, 1, 1, 1], padding='SAME', name='mconv2')
    sconv1 = tf.nn.conv2d(tf.transpose(screen, [0, 2, 3, 1]), screen_filters1, strides=[1, 1, 1, 1], padding='SAME', name='sconv1')
    sconv2 = tf.nn.conv2d(sconv1, screen_filters2, strides=[1, 1, 1, 1], padding='SAME', name='sconv2')
    info_fc = layers.fully_connected(layers.flatten(info), num_outputs=256, activation_fn=tf.tanh, scope='info_fc')

    flat_screen = tf.reshape(sconv2, [-1, 16*16*32])
    dense_screen = tf.layers.dense(inputs=flat_screen, units=1024, activation=tf.nn.relu)
    self.screen_output = tf.layers.dense(dense_screen, 256)

    flat_minimap = tf.reshape(mconv2, [-1, 16*16*32])
    dense_minimap = tf.layers.dense(inputs=flat_minimap, units=1024, activation=tf.nn.leaky_relu)
    self.minimap_output = tf.layers.dense(dense_minimap, 64)

    self.saver = tf.train.Saver() # define a saver for saving and restoring
    self.writer = tf.summary.FileWriter('./action_and_id_log', self.sess.graph)     # write to file
    self.merge_op = tf.summary.merge_all() # operation to merge all summary

  def build_opt(self):
    l1_user_info = tf.layers.dense(self.info, self.isize, tf.tanh)
    user_info_output = tf.layers.dense(l1_user_info, 5)

    # regression, NOT SURE IF THIS IS suitable regression (?)
    input_to_classification = tf.concat([self.minimap_output, self.screen_output, user_info_output], 1)  # state rep? this shits a mess
    l2_classification = tf.layers.dense(input_to_classification, 1024, tf.nn.relu)
    classification_output = tf.layers.dense(l2_classification, 543)              # output layer
    self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.action_output, logits=classification_output)

    self.train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    tf.summary.scalar('loss', loss)

    self.accuracy = tf.metrics.accuracy(       # creates local vars (?)
        labels=tf.argmax(self.action_output, axis=1), predictions=tf.argmax(classification_output, axis=1),)[1]

    # should this be here?
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
    self.sess.run(init_op) # initialize var in graph
    self.opt_op()  # could put this here for now. Once batchsize is a thing this will have to go elsewhere

  def opt_op(self):
    print("Beginning training session")
    for step in range(250):
        m,s,u,a,params =  self.get_batch(get_action_id_only=True)
        self.logger.info(f"minimap: {m}")
        self.logger.info(f"len(minimap): {len(m)}")
        feed_dict = {
            self.minimap: m,
            self.screen: s,
            self.info:u,
            self.action_output: a}
        # param handling if need be
        # have to handle writer and saver shit too.... check out baselines.utils they had a clean way  ***
        _, loss_, result = self.sess.run([self.train_op, self.loss, self.merge_op], feed_dict=feed_dict)
        self.writer.add_summary(result, step)

        if step % 50 == 0:
            accuracy_ = self.sess.run([self.accuracy],
                {self.minimap: m,
                self.screen: s,
                self.info:u,
                self.action_output: a})
            print('Step:', step, '| train loss: ', loss_, '| test accuracy: ', accuracy_)


    self.saver.save(sess, './params', write_meta_graph=False)  # meta_graph is not recommended

  def _init_logger(self, dir):
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(dir, mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    self.logger.addHandler(file_handler)

