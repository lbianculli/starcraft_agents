from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
import tensorflow as tf
import tensorflow.contrib.layers as layers
import utils as U


SCREEN_TYPES = [actions.TYPES[0], actions.TYPES[2]]
MINIMAP_TYPES = [actions.TYPES[1]]


def build_net(minimap, screen, info, msize, ssize, num_action):
  # Extract features
  screen_filters1 = tf.get_variable(name='sf1', shape=(5, 5, U.screen_channel(), 16))  # hwio
  screen_filters2 = tf.get_variable(name='sf2',shape=(3, 3, 16, 32))
  minimap_filters1 = tf.get_variable(name='mmf1',shape=(5, 5, U.minimap_channel(), 16))
  minimap_filters2 = tf.get_variable(name='mmf2',shape=(3, 3, 16, 32))

  mconv1 = tf.nn.conv2d(tf.transpose(minimap, [0, 2, 3, 1]), minimap_filters1, strides=[1, 1, 1, 1], padding='SAME', name='mconv1')
  mconv2 = tf.nn.conv2d(mconv1, minimap_filters2, strides=[1, 1, 1, 1], padding='SAME', name='mconv2')
  sconv1 = tf.nn.conv2d(tf.transpose(screen, [0, 2, 3, 1]), screen_filters1, strides=[1, 1, 1, 1], padding='SAME', name='sconv1')
  sconv2 = tf.nn.conv2d(sconv1, screen_filters2, strides=[1, 1, 1, 1], padding='SAME', name='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info), num_outputs=256, activation_fn=tf.tanh, scope='info_fc')

  # Compute spatial actions
  feat_conv = tf.concat([mconv2, sconv2], axis=3)

  spatial_weights = tf.get_variable(name='spatial_weights', shape=(1, 1, feat_conv.get_shape()[-1], 1))
  spatial_action = tf.nn.conv2d(feat_conv, spatial_weights, strides=[1, 1, 1 ,1], padding='SAME', name='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

  # Compute non spatial actions and value
  state_representation = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  state_representation = layers.fully_connected(state_representation, num_outputs=256, activation_fn=tf.nn.relu, scope='state_rep')
  non_spatial_action = layers.fully_connected(state_representation, num_outputs=num_action, activation_fn=tf.nn.softmax, scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(state_representation, num_outputs=1, activation_fn=None, scope='value'), [-1])


  return spatial_action, non_spatial_action, state_representation, value
