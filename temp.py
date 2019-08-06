
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features


_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_MINIMAP_FEATURES = 5
_SCREEN_FEATURES = 8


def preprocess_minimap(minimap):
  layers = []
  assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
  # assert minimap.shape[0] == _MINIMAP_FEATURES

  for i in range(len(features.MINIMAP_FEATURES)):
  # for i in range(_MINIMAP_FEATURES):
    if i == _MINIMAP_PLAYER_ID:                                               # if the feature is player_id
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)      # normalize
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:    # it it's a scalar valued feature (other is categorical)
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)      # normalize -- why not put these in same line?
    else:                                                                     # if categorical feature
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale,                   # create array of zeroes (scale, px1, px2)  --> ex: (17, 64, 64)
        minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):                     # for j in range of scale value (e.g: creep is 2)
        indy, indx = (minimap[i] == j).nonzero()                              # where minimap corresponds to that value not being 0
        layer[j, indy, indx] = 1                                              # encode that element in the tensor
      layers.append(layer)
  return np.concatenate(layers, axis=0)                                       # concat all layers (one for each feature)


def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  # assert screen.shape[0] == 8

  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    else:
      layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(features.SCREEN_FEATURES[i].scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def minimap_channel():
  c = 0
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      c += 1
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.MINIMAP_FEATURES[i].scale
  return c


def screen_channel():
  c = 0
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      c += 1
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.SCREEN_FEATURES[i].scale
  return c


def clean_dir(parent_path, verbose=True, excluded_paths=[]):
    '''
    Given a directory, through and deletes all files from the topdown
    Can specify files to save as a list
    '''
    for (root, dirs, files) in os.walk(parent_path, topdown=True):
        if len(files) > 0:
            for file in files:
                child_path = root + '/' +  file

                if os.path.isfile(child_path) and child_path not in excluded_paths:
                    os.unlink(child_path)
                    if verbose:
                        print(f'{child_path} deleted')


class SessionManager:
  def __init__(self, sess=None, summary_writer=None, base_path="results/", ckpt_freq=100, training_enabled=True):
    if not sess:
      config = tf.ConfigProto()
      config.allow_soft_placement = True
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    self.sess = sess
    self.base_path = base_path
    self.ckpt_freq = ckpt_freq
    self.training_enabled = training_enabled
    self.global_step = tf.train.get_or_create_global_step()
    if not summary_writer:
      self.summary_writer = tf.summary.FileWriter(self.summaries_path)

  def restore_or_init(self):
    """ creates saver, loads from checkpoint if one exists. otherwise setup graph """
    self.saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(self.checkpoint_path)  # ?
    if ckpt:
      self.saver.restore(self.sess, ckpt)

      if self.training_enabled:
        # merge with previous summary session -- will this let global step load as well?
        self.summary_writer.add_session_log(  # not sure what these do?
          tf.SessionLog(status=tf.SessionLog.START), self.sess.run(self.global_step))

    else:
      self.sess.run(tf.global_variables_initializer())
    # this call locks the computational graph into read-only state,
    # as a safety measure against memory leaks caused by mistakingly adding new ops to it
    self.sess.graph.finalize()

  def run(self, tf_op, tf_inputs=None, inputs=None, feed_dict=None):
    if feed_dict:
      return self.sess.run(tf_op, feed_dict=feed_dict)

    return self.sess.run(tf_op, feed_dict=dict(zip(tf_inputs, inputs)))  # pretty handy

  # def on_update(self, step):
  #   """ saves checkpoint if appropriate """
  #   if not self.ckpt_freq or not self.training_enabled or step % self.ckpt_freq:
  #     return
  #   self.saver.save(self.sess, self.checkpoint_path + "/.ckpt", global_step=step)

  def _save(self, path=None, count=None):
    self.saver.save(self.sess, path+'/model.pkl', count)  # note: episode count

  def add_summaries(self, tag, value, prefix='', step=None):
    if not self.training_enabled:
      return
    summary = self.create_summary(prefix + "/" + tag, value)
    self.summary_writer.add_summary(summary, global_step=step)

  @staticmethod  # remember: staticmethod is ...
  def create_summary(tag, value):  # not sure any of this
    """ creates summary """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

  @property # remember: property .... is used for...
  def start_step(self):
    if self.training_enabled:
      return self.global_step.eval(session=self.sess)
    return 0

  @property
  def summaries_path(self):
    return self.base_path + "/summaries"

  @property
  def checkpoint_path(self):
    return self.base_path + "/checkpoints"
