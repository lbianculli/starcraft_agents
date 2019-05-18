import numpy as np
import tensorflow as tf
from dqn_utils import huber_loss

class PlayerRelativeMovementCNN():
    ''' uses feature_screen.player_relative to assign q value to movements '''

    def __init__(self,
                 spatial_dims,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name='DQN'):
        ''' initialize instance-specific hyperparams, build tf graph '''

        self.spatial_dims = spatial_dims
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.summary_path = summary_path
        self.name = name

        self._build()
        if self.summary_path:
            # setup summary saver
            self.writer = tf.summary.FileWriter(self.summary_path)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('score', self.score)
            tf.summary.scalar('mean_q', self.mean_q)
            tf.summary.scalar('max_q', self.max_q)
            self.write_op = tf.summary.merge_all()

        if self.save_path:
            # setup model saver
            self.saver = tf.train.Saver()


    def save_model(self, sess):
        ''' Write TF checkpoint '''
        self.saver.save(sess, self.save_path)


    def load(self, sess, ckpt=None):
        ''' Load from TF checkpoint '''
        if ckpt is None:
            self.saver.restore(sess, self.save_path)
        else:
            self.saver.restore(sess, ckpt)

        tf.global_variables_initializer()



    def write_summary(self, sess, states, actions, targets, score):
        ''' Write session summary to TensorBoard '''
        global_episode = self.global_episode.eval(sess)  # what is global_episode at runtime? -- variable with shape 0
        summary = sess.run(
                    self.write_op,
                    feed_dict={self.inputs: states,
                               self.actions: actions,
                               self.targets: targets,
                               self.score: score})

        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush()



    def increment_global_episode_op(self, sess):
        ''' Increment the global episode tracker '''
        sess.run(self.increment_global_episode)


    def optimizer_op(self, sess, states, actions, targets, ret=False):
        ''' Perform one iteration of gradient updates '''
        loss, _ = sess.run([self.loss, self.optimizer],
                           feed_dict={self.inputs: states,
                                      self.actions: actions,
                                      self.targets: targets})

        if ret:
            return loss



    def _activation_summary(self, x):
        ''' Creates activation summaries for TB. '''
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(x))


    def _build(self):
        # this seems to be mostly the same, although doesnt follow paper specs exactly...
        """Construct graph."""
        with tf.variable_scope(self.name) as scope:
            # score tracker
            self.score = tf.placeholder(tf.int32, [], name="score")

            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.global_episode = tf.Variable(0, trainable=False, name="global_episode")

            # placeholders
            self.inputs = tf.placeholder(tf.int32,[None, *self.spatial_dims], name="inputs")  # [None, px1, px2]
            self.actions = tf.placeholder(tf.float32, [None, np.prod(self.spatial_dims)], name="actions")
            self.targets = tf.placeholder(tf.float32, [None], name="targets")

            # increment counts
            self.increment_global_episode = tf.assign(self.global_episode, self.global_episode + 1, name="increment_global_episode")
            self.increment_global_step = tf.assign(self.global_step, self.global_step + 1, name="increment_global_step")

            # NOTE: spatial coordinates are given in y-major screen coordinate space
            self.transposed = tf.transpose(self.inputs, perm=[0, 2, 1], name="transpose")
            self.one_hot = tf.one_hot(self.transposed, depth=5, axis=-1, name="one_hot")  
            embed_filters = tf.get_variable(name='embed_filters', shape=[64, 64, 5, 1])  # [None, 64, 64, 1]
            self.embed = tf.nn.conv2d(self.one_hot, embed_filters, strides=[1, 1, 1, 1], padding="SAME", name="embed") 

            conv1_filters = tf.get_variable(name='layer1_filters', shape=[64, 64, 1, 16])
            self.conv1 = tf.nn.conv2d(self.embed, conv1_filters, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
            self.conv1_act = tf.nn.relu(self.conv1, name='conv1_act')
            self._activation_summary(self.conv1_act)

            output_filters = tf.get_variable(name='output_filters', shape=[64, 64, 16, 1])  # output will be scalar
            self.output = tf.nn.conv2d(self.conv1_act, output_filters, strides=[1, 1, 1, 1], padding='SAME', name='output')

            self.flat = tf.layers.flatten(self.output, name='flat')  # ex: [None, 4, 4] --> [None, 16]

            # value estimate trackers for summaries
            self.max_q = tf.reduce_max(self.flat, name='max_q')
            self.mean_q = tf.reduce_mean(self.flat, name='mean_q')  # if axis=None, reduce along all dims (scalar)

            # optimization: MSE between state predicted Q and target Q
            self.prediction = tf.reduce_sum(tf.multiply(self.flat, self.actions), axis=1, name='preds')
            self.loss = tf.reduce_mean(tf.square(self.targets - self.prediction), name='loss') 
            self.loss2 = tf.square(self.targets-self.prediction, name='loss2')  # element wise for priority updates
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
