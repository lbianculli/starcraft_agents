from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
import utils as U


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
  
  argument_policy = dict()
  arguments = dict()
  for arg_type in actions.TYPES:

      # for spatial actions, represent each dimension independently
      if len(arg_type.sizes) > 1:
          if arg_type in SCREEN_TYPES:
              units = ssize
          elif arg_type in MINIMAP_TYPES:
              units = msize

          arg_policy_x = layers.fully_connected(
              inputs=state_representation,
              units=units,
              activation=tf.nn.softmax)

          arg_policy_y = layers.fully_connected(
              state_representation,
              units,
              activation=tf.nn.softmax)

          argument_policy[str(arg_type) + "x"] = arg_policy_x
          argument_policy[str(arg_type) + "y"] = arg_policy_y

          arg_placeholder_x = tf.placeholder(
              tf.float32,
              shape=[None, units])

          arg_placeholder_y = tf.placeholder(
              tf.float32,
              shape=[None, units])

          arguments[str(arg_type) + "x"] = arg_placeholder_x
          arguments[str(arg_type) + "y"] = arg_placeholder_y

      else:
          arg_policy = layers.fully_connected(state_representation,
              num_outputs=arg_type.sizes[0],
              activation=tf.nn.softmax)

          argument_policy[str(arg_type)] = arg_policy

          arg_placeholder = tf.placeholder(
              tf.float32,
              shape=[None, arg_type.sizes[0]])

          arguments[str(arg_type)] = arg_placeholder

  return spatial_action, non_spatial_action, state_representation, value, arguments, argument_policy

