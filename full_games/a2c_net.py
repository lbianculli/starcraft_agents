import numpy as np
import tensorflow as tf
from preprocessing import preprocess_spatial_features
from pysc2.lib import actions, features

SCREEN_FEATURES = features.SCREEN_FEATURES
MINIMAP_FEATURES = features.MINIMAP_FEATURES
SCREEN_DIMS = [32, 32]
MINIMAP_DIMS = [32, 32]
NUM_ACTIONS = len(actions.FUNCTIONS)  # so num_actions is total possible actions

# manually state the argument types which take points on screen/minimap
SCREEN_TYPES = [actions.TYPES[0], actions.TYPES[2]]  # [a point on the screen, second point for a rectangle]
MINIMAP_TYPES = [actions.TYPES[1]]  # [point on the minimap]

class AlphaCNN():
    def __init__(self,
                 screen_dims=SCREEN_DIMS,
                 minimap_dims=MINIMAP_DIMS,
                 learning_rate=1e-4,
                 value_gradient_strength=.5,  # check these
                 regularization_strength=.01,
                 screen_features=17,  # obs.observation.feature_screen?
                 minimap_features=7,  # obs.observation.feature_minimap? do i need this here? dont think so
                 save_path=None,
                 summary_path=None,
                 name='ac_net'):

    # input
        tf.reset_default_graph()
        self.screen_dims = screen_dims
        self.minimap_dims = minimap_dims
        self.learning_rate = learning_rate
        self.value_gradient_strength = value_gradient_strength
        self.regularization_strength = regularization_strength
        self.screen_features = screen_features
        self.minimap_features = minimap_features
        self.save_path = save_path
        self.summary_path = summary_path

        with tf.variable_scope(name):
            self._build()
            self._build_optimization()

        # setup model saver
        if self.summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Policy_Loss", self.policy_loss)
            tf.summary.scalar("Value_Loss", self.value_loss)
            tf.summary.scalar("Entropy", self.entropy)
            tf.summary.scalar("A2C_Loss", self.total_loss)
            self.write_op = tf.summary.merge_all()
            self.logger.info(f'To view Tensorboard, run tensorboard --logdir={summary_path}')

        if self.save_path:
            self.saver = tf.train.Saver()


    def _activation_summary(self, tensor):
        ''' Creates summaries for activations. Returns nothing. '''
        tensor_name = tensor.op.name
        tf.summary.histogram(tensor_name + '/activations', tensor)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(tensor))


    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)


    def write_summary(self, sess, global_episode, score, feed_dict):
        """Write summary to Tensorboard."""
        feed_dict[self.score] = score

        summary = sess.run(
            self.write_op,
            feed_dict=feed_dict)
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.add_graph(sess)
        self.writer.flush


    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)


    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)


    def optimizer_op(self, sess, feed_dict):
        """Perform one iteration of gradient updates."""
        sess.run(self.optimizer, feed_dict=feed_dict)


    def _build(self):
        ''' Create network architecture for agent '''
        self.score = tf.placeholder(tf.int32, [], name='score')
        self.global_step = tf.Variable(0,trainable=False,name='global_step')
        self.global_episode = tf.Variable(0,trainable=False,name='global_episode')
        self.increment_global_episode = tf.assign(self.global_episode, self.global_episode + 1,
                                                  name='increment_global_episode')

        self.inputs = tf.placeholder(tf.int32,[None, *self.screen_dims], name="inputs")  # [None, px1, px2]
        self.actions = tf.placeholder(tf.float32, [None, np.prod(self.screen_dims)], name='actions')
        self.targets = tf.placeholder(tf.float32, [None], name="targets")  # features.Player is broadcast vector.

        self.screen_inputs = tf.placeholder(tf.int32, shape=[None, len(SCREEN_FEATURES), *self.screen_dims], name='screen_inputs')
        self.minimap_inputs = tf.placeholder(tf.int32, shape=[None, len(MINIMAP_FEATURES), *self.minimap_dims], name='minimap_inputs')
        self.flat_inputs = tf.placeholder(tf.float32, shape=[None, len(features.Player)], name='flat_inputs')  # env stuff: mins, gas, supply, etc -- 11 total
        self.screen_preprocessed = preprocess_spatial_features(self.screen_inputs, screen=True)
        self.minimap_preprocessed = preprocess_spatial_features(self.minimap_inputs, screen=False)
        self.flat_preprocessed = tf.log(self.flat_inputs + 1., name='flat_preprocessed')

        # onto conv net. filter variables first
        screen_filters1 = tf.get_variable(name='screen_f1', shape=(5, 5, self.screen_preprocessed.shape[-1], 16))  # hwio
        screen_filters2 = tf.get_variable(name='screen_f2', shape=(3, 3, 16, 32))  # anything else for these?
        minimap_filters1 = tf.get_variable(name='minimap_f1', shape=(5, 5, self.minimap_preprocessed.shape[-1], 16))
        minimap_filters2 = tf.get_variable(name='minimap_f2', shape=(3, 3, 16, 32))

        screen_conv1 = tf.nn.conv2d(self.screen_preprocessed, screen_filters1, strides=[1, 1, 1, 1], padding='SAME', name='screen_conv1_in')
        screen_conv1 = tf.nn.relu(screen_conv1, name='screen_conv1_out')
        screen_conv2 = tf.nn.conv2d(screen_conv1, screen_filters2, strides=[1, 1, 1, 1], padding='SAME', name='screen_conv2_in')
        screen_conv2 = tf.nn.relu(screen_conv2, name='screen_conv2_out')
        self._activation_summary(screen_conv2)

        minimap_conv1 = tf.nn.conv2d(self.minimap_preprocessed, minimap_filters1, strides=[1, 1, 1, 1], padding='SAME', name='minimap_conv1_in')
        minimap_conv1 = tf.nn.relu(minimap_conv1, name='minimap_conv1_out')
        minimap_conv2 = tf.nn.conv2d(minimap_conv1, minimap_filters2, strides=[1, 1, 1, 1], padding='SAME', name='minimap_conv2_in')
        minimap_conv2 = tf.nn.relu(minimap_conv2, name='minimap_conv2_out')
        self._activation_summary(minimap_conv2)

        # linear layer for non-spatial features (tanh activation)
        flat_linear = tf.layers.dense(self.flat_preprocessed, units=64, activation=tf.nn.tanh, name='flat_linear')
        self._activation_summary(flat_linear)

        # flatten layers and concat. Note: flat_linear is [None, 64]
        screen_flat = tf.layers.flatten(screen_conv2, name='screen_flat')
        minimap_flat = tf.layers.flatten(minimap_conv2, name='minimap_flat')
        concat_layer = tf.concat([screen_flat, minimap_flat, flat_linear], axis=1, name='concat_layer')

        self.state_representation = tf.layers.dense(concat_layer, 256, activation=tf.nn.relu, name='state_rep')
        self.policy = tf.squeeze(tf.layers.dense(
            inputs=self.state_representation,
            units=NUM_ACTIONS,
            activation=tf.nn.softmax,
            name='policy'))

        self._activation_summary(self.state_representation)
        self._activation_summary(self.policy)

        # spatial vs. non spatial starts down here
        # action function argument policies (nonspatial)
        # action function argument placeholders (for optimization)
        self.argument_policy = dict()
        self.arguments = dict()

        for arg_type in actions.TYPES:
            # for spatial actions, represent each dimension independently
            if len(arg_type.sizes) > 1:
                if arg_type in SCREEN_TYPES:
                    units = self.screen_dims
                elif arg_type in MINIMAP_TYPES:
                    units = self.minimap_dims  # so these will be [n, n]

                # get x, y probs
                arg_policy_x = tf.layers.dense(self.state_representation, units=units[0], activation=tf.nn.softmax)
                arg_policy_y = tf.layers.dense(self.state_representation, units=units[1], activation=tf.nn.softmax)

                self.argument_policy[str(arg_type) + 'x'] = arg_policy_x  # prob of each x
                self.argument_policy[str(arg_type) + 'y'] = arg_policy_y  # prob of each y

                arg_ph_x = tf.placeholder(tf.float32, [None, units[0]])
                arg_ph_y = tf.placeholder(tf.float32, [None, units[1]])

                self.arguments[str(arg_type) + 'x'] = arg_ph_x
                self.arguments[str(arg_type) + 'y'] = arg_ph_y
            else:
                arg_policy = tf.layers.dense(self.state_representation, units=arg_type.sizes[0], activation=tf.nn.softmax)
                self.argument_policy[str(arg_type)] = arg_policy

                arg_ph = tf.placeholder(tf.float32, [None, arg_type.sizes[0]])
                self.arguments[str(arg_type)] = arg_ph


    def _build_optimization(self):  # "params learnt with AC"
        ''' construct a graph for network updates '''
        self.actions = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name='actions_ph')
        self.reward = tf.placeholder(tf.float32, shape=(None), name='reward_ph')  # actual reward

        self.action_prob = tf.reduce_sum(self.policy * self.actions, axis=1, name='action_prob_ph')
        self.args_prob = 1.
        for arg_type in self.arguments:  # arg_type: placeholder where placeholder is coordinate(s)
            # compute probability for each argument
            arg_probability = tf.reduce_sum(self.arguments[arg_type] * self.argument_policy[arg_type])
            nonzero_probs = tf.cond(
                tf.logical_not(tf.equal(arg_probability, 0)),
                true_fn=lambda: arg_probability,
                false_fn=lambda: 1.)

            self.args_prob *= nonzero_probs

        self.value_estimate = tf.layers.dense(
            self.state_representation, units=1, activation=None, name='value_estimate')

        self.advantage = tf.subtract(
            self.reward, tf.squeeze(tf.stop_gradient(self.value_estimate)), name='advantage')

        # ac gradient = policy gradient + value gradient + entropy
        self.policy_loss = -tf.reduce_mean(
            (self.advantage * tf.log(self.action_prob * self.args_prob)), name='policy_loss')
        self.value_loss = -tf.reduce_mean(
            self.advantage * tf.squeeze(self.value_estimate), name='value_loss')

        # only including function identifier entropy, not args
        self.entropy = tf.reduce_sum(self.policy * tf.log(self.policy), name='entropy')

        self.total_loss = tf.add_n(
            inputs=[self.policy_loss,
            self.value_loss * self.value_gradient_strength,
            self.entropy * self.regularization_strength],
            name='total_loss')

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss, global_step=self.global_step)
