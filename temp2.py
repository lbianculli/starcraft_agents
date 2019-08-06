import os
import numpy as np
import pickle
from google.protobuf.json_format import MessageToJson
import json
from pysc2.lib import actions
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("C:/Users/lbianculli/action_param_log", mode="w")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class BatchGenerator(object):
	def __init__(self):
		self.home_dir = "C:/Users/lbianculli"
		self.parsed_directory = self.home_dir+'/agent_replay_data/'
		self.parsed_filenames = os.listdir(self.parsed_directory)  # list of pickled replays
		self.next_index = 0
		self.dimension = 64  #
    self.argument_policy = dict()
    self.arguments = dict()

	# every batch corresponding to 1 replay file
	def next_batch(self, get_action_id_only=False):
		""" Loads pickled replay data. Returns screen, minimap, actions, player info and coords. For supervised learning """
		full_filename = self.parsed_directory + self.parsed_filenames[self.next_index]  # single replay path
		# while os.stat(full_filename).st_size == 0:
		while os.path.getsize(full_filename) == 0:
			del self.parsed_filenames[self.next_index]
			full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
			if self.next_index >= len(self.parsed_filenames):
				self.next_index = 0

		self.next_index += 1
		if self.next_index == len(self.parsed_filenames):
			self.next_index = 0

		try:
			replay_data = pickle.load(open(full_filename, "rb"))
		except:
			return self.next_batch(get_action_id_only)

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
			return self.next_batch(get_action_id_only)

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
			m_temp = np.array(state['minimap'])
			m_temp = np.reshape(m_temp, [self.dimension, self.dimension,5])  # this is where shape error is originating
			s_temp = np.array(state['screen'])
			s_temp = np.reshape(s_temp, [self.dimension,self.dimension,10])
			
			# one-hot action_id
			last_action = None
			for action in state['actions']:
				if last_action == action:
					continue

				one_hot = np.zeros((1, 543)) # Not sure where 543 is from. Hardcoding for now
				one_hot[np.arange(1), [action[0]]] = 1

				for param in action[2]:  # FUNCTIONS[func_id].name --> ex: "Build_factory_screen", "Attack_screen"
					if param == [0]:  # seen this before. idx corresponds to action id (no_op here). Should i have this if statement?
						continue
					minimap_output.append(m_temp)
					screen_output.append(s_temp)
					action_output.append(one_hot[0])
					player_info_output.append(pi_temp)
# 					ground_truth_coordinates.append(np.array(param))  # would need to introduce some handling he. to what extent?

				  if len(param) > 1:
				    # handle for non-spatial actions
				    if arg_type in SCREEN_TYPES:
					units = self.screen_dimensions  # 64
				    elif arg_type in MINIMAP_TYPES:
					units = self.minimap_dimensions  # 64

				    arg_policy_x = tf.layers.dense(
					inputs=self.state_representation,
					units=units[0],
					activation=tf.nn.softmax)

				    arg_policy_y = tf.layers.dense(
					inputs=self.state_representation,
					units=units[1],
					activation=tf.nn.softmax)

				    self.argument_policy[str(arg_type) + "x"] = arg_policy_x
				    self.argument_policy[str(arg_type) + "y"] = arg_policy_y

				    arg_placeholder_x = tf.placeholder(
					tf.float32,
					shape=[None, units[0]])

				    arg_placeholder_y = tf.placeholder(
					tf.float32,
					shape=[None, units[1]])

				    self.arguments[str(arg_type) + "x"] = arg_placeholder_x
				    self.arguments[str(arg_type) + "y"] = arg_placeholder_y

				else:
				  arg_policy = tf.layers.dense(
				      inputs=self.state_representation,
				      units=arg_type.sizes[0],
				      activation=tf.nn.softmax)

				  self.argument_policy[str(arg_type)] = arg_policy

				  arg_placeholder = tf.placeholder(
				      tf.float32,
				      shape=[None, arg_type.sizes[0]])

				  self.arguments[str(arg_type)] = arg_placeholder


					assert(len(minimap_output) == len(ground_truth_coordinates))

					if len(minimap_output) == 0:
						# The replay file only record one person's operation, so if it is 
						# the defeated person, we need to skip the replay file
						return self.next_batch(get_action_id_only)

					if get_action_id_only:
						return minimap_output, screen_output, player_info_output, action_output
			# 		else:
			# 			return minimap_output, screen_output, action_output, player_info_output, ground_truth_coordinates
					else:
						return minimap_output, screen_output, action_output, player_info_output, self.arguments, self.argument_policy  
			    # then need to handle in SL file

			
			
			
# -------------- SL FILE BELOW --------------- #
# ** this is really jumbled. I would rather consolidate these into a class. would look very similar to ray's
import tensorflow as tf
import numpy as np
from batch_generator import BatchGenerator
import logging
import utils as U
from pysc2.lib import actions
import tensorflow.contrib.layers as layers


SCREEN_TYPES = [actions.TYPES[0], actions.TYPES[2]]
MINIMAP_TYPES = [actions.TYPES[1]]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("C:/Users/lbianculli/action_and_id_log", mode="w")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

isize = 11
msize= 64
ssize = 64

score = tf.placeholder(tf.int32, [], name='score')
minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), msize, msize], name='minimap')  # 17, 64, 64
screen = tf.placeholder(tf.float32, [None, U.screen_channel(), ssize, ssize], name='screen')
info = tf.placeholder(tf.float32, [None, isize], name='info')

# minimap_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 5])
# screen_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 10])
# user_info_placeholder = tf.placeholder(tf.float32, [None, isize])
action_output = tf.placeholder(tf.float32, [None, 543]) # one hot

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
screen_output = tf.layers.dense(dense_screen, 256)

flat_minimap = tf.reshape(mconv2, [-1, 16*16*32])   
dense_minimap = tf.layers.dense(inputs=flat_minimap, units=1024, activation=tf.nn.leaky_relu)
minimap_output = tf.layers.dense(dense_minimap, 64)

# user info
l1_user_info = tf.layers.dense(info, isize, tf.tanh)
user_info_output = tf.layers.dense(l1_user_info, 5)

# regression, NOT SURE IF THIS IS suitable regression
input_to_classification = tf.concat([minimap_output, screen_output, user_info_output], 1)

l2_classification = tf.layers.dense(input_to_classification, 1024, tf.nn.relu)
classification_output = tf.layers.dense(l2_classification, 543)              # output layer
loss = tf.losses.softmax_cross_entropy(onehot_labels=action_output, logits=classification_output)

train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
tf.summary.scalar('loss', loss) # add loss to scalar summary

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(action_output, axis=1), predictions=tf.argmax(classification_output, axis=1),)[1]

# setup for graph
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True   
sess = tf.Session(config=config)                                # control training and others
# sess.run(tf.global_variables_initializer(), tf.local_variables_initializer())    # initialize var in graph
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op) # initialize var in graph

saver = tf.train.Saver() # define a saver for saving and restoring
writer = tf.summary.FileWriter('./action_and_id_log', sess.graph)     # write to file
merge_op = tf.summary.merge_all() # operation to merge all summary

bg = BatchGenerator()
print("Beginning training session")
for step in range(250):                             # train
    m,s,u,a =  bg.next_batch(get_action_id_only=True)
    _, loss_, result = sess.run([train_op, loss, merge_op],
        {minimap: m, 
        screen: s, 
        info:u,
        action_output: a})
    writer.add_summary(result, step)

    if step % 50 == 0:
        accuracy_ = sess.run([accuracy],
            {minimap: m, 
            screen: s, 
            info:u,
            action_output: a})
        print('Step:', step, '| train loss: ', loss_, '| test accuracy: ', accuracy_)

    print('---------------------------------------------------')

saver.save(sess, './params', write_meta_graph=False)  # meta_graph is not recommended






				
