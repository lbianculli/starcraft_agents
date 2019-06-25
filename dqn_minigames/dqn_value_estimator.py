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
            # tf.summary.scalar('loss', self.loss)  # not gonna work if using huber loss
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


    def _init_train_fn(self, var_list, grad_norm_clipping=10):
        self.train_fn = self._minimize_and_clip(self.optimizer, self.loss,  var_list=var_list, clip_val=grad_norm_clipping)


    def optimizer_op(self, sess, states, actions, targets, var_list) :
        '''
        Perform one iteration of gradient updates.
        Has to be a way to make this more dynamic
        '''
        loss, _ = sess.run([self.loss, self.train_fn], feed_dict={
            self.inputs: states,
            self.actions: actions,
            self.targets: targets
            })  # run at the end. literally exact same as regular


    def _minimize_and_clip(self, optimizer, objective, var_list, clip_val=10):
        """Minimized `objective` using `optimizer` w.r.t. variables in
        `var_list` while ensure the norm of the gradients for each
        variable is clipped to `clip_val`
        """
        gradients = optimizer.compute_gradients(objective, var_list=var_list)  # will need to do this in other dqn_move_only
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients, global_step=self.global_step)


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
            # if going to increment global step, should be done every time i run thru optimizer

            # spatial coordinates are given in y-major screen coordinate space
            # transpose them to (x, y) space before beginning # filter: hwio; input: nhwc
            self.transposed = tf.transpose(self.inputs, perm=[0, 2, 1], name="transpose")
            self.one_hot = tf.one_hot(self.transposed, depth=5, axis=-1, name="one_hot")  # [None, 64, 64, 5] -- one hot the categorical features. why depth 5?

            embed_filters = tf.get_variable(name='embed_filters', shape=[64, 64, 5, 1])  # [None, 64, 64, 1]
            self.embed = tf.nn.conv2d(self.one_hot, embed_filters, strides=[1, 1, 1, 1], padding="SAME", name="embed") # pass thru 1x1 conv

            # network architecture: using only one conv to start, can add later
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

            # optimization: MSE between state predicted Q and target Q ### *** NEED TO CLEAN THIS UP --> consolidate losses and opts
            self.prediction = tf.reduce_sum(tf.multiply(self.flat, self.actions), axis=1, name='preds')
            self.loss = huber_loss(self.targets-self.prediction)  # element wise for priority updates
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)  # why no minimize?
