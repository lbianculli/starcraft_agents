import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Concatenate, Dense, Embedding, Conv2D, Flatten, Lambda
from my_layers import Squeeze, Split, Transpose, Log, Broadcast2D


def build_fully_conv(obs_spec, act_spec, data_format="channels_first", broadcast_non_spatial=False, fc_dim=256):





def spatial_block(name, space, cfg):
  inpt = Input(space.shape, name=name + "_input")
  block = Split(space.shape[0], axis=1)(inpt)

  for i, (name, dim) in enumerate(zip(space.spatial_feats, space.spatial_dims)):  # what are these? Need to find where these are called
    if dim > 1:
      block[i] = Squeeze(axis=1)(block[i])
      block[i] = Embedding(input_dim=dim, output_dim=10)(block[i])
      # [N, H, W, C] -> [N, C, H, W]
      block[i] = Transpose([0, 3, 1, 2])(block[i])
    else:
      block[i] = Log()(block[i])  # looks familiar

  block = Concatenate(axis=1)(block)
  block = Conv2D(16, 5, **cfg)(block)
  block = Conv2D(32, 3, **cfg)(block)

  return block, inpt



def conv_cfg(data_format='channels_first', activation=None, scale=1.0):
  return dict(
      padding='same',
      activation=activation,
      data_format=data_format,
      kernel_initializer=VarianceScaling(scale=2.0*scale)  # what's VarianceScaling?
  )


def dense_cfg(activation=None, scale=1.0):
  return dict(
      activation=activation,
      kernel_initializer=VarianceScaling(scale=2.0*scale)
  )
