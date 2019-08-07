import tensorflow as tf
import numpy as np
from batch_generator import BatchGenerator
import logging
import utils as U
from pysc2.lib import actions
import tensorflow.contrib.layers as layers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("C:/Users/lbianculli/action_param_log", mode="w")
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
param_output = tf.placeholder(tf.float32, [None, 2], name="params")

# minimap_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 5])
# screen_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 10])
# user_info_placeholder = tf.placeholder(tf.float32, [None, isize])
action_output = tf.placeholder(tf.float32, [None, 543]) # one hot

# set up network
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
sess.run(init_op) 

saver = tf.train.Saver() # define a saver for saving and restoring
writer = tf.summary.FileWriter('./action_param_log', sess.graph)     # write to file
merge_op = tf.summary.merge_all() # operation to merge all summary


bg = BatchGenerator()
for step in range(1000):                             # train. does number matter?
    m,s,a,u,y =  bg.next_batch()
    logger.info(f"Y: {y[:20]}\n--------------------------------------\n")  # all length 1732. all lists of arrays
    logger.info(f"screen: {s[:10]}\n--------------------------------------\n")
    logger.info(f"action: {a[:10]}\n--------------------------------------\n")
    _, loss_, result = sess.run([train_op, loss, merge_op],
        {minimap: m, 
        screen: s, 
        action_output: a, 
        info:u, 
        param_output:y})  # y is issue, ground_truth_coordinates. I think the issue is some returning arrays of len(1). Non-spatial?
    writer.add_summary(result, step)
    print('step: ', step, 'loss: ',loss_, 'result: ', result)

saver.save(sess, './params', write_meta_graph=False)  # meta_graph is not recommended





