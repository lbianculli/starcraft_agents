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
                 value_gradient_strength=.1,  # check these
                 regularization_strength=.05,
                 screen_features=17,  # obs.observation.feature_screen?
                 minimap_features=7,  # obs.observation.feature_minimap? do i need this here? dont think so
                 save_path=None,
                 summary_path=None,
                 name='full_game_CNN'):

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
        if self.save_path:
            self.saver = tf.train.Saver()
            self.write_op = tf.summary.merge_all()


    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)


    def write_summary(self, sess, feed_dict):
        ''' Write session summary to TensorBoard '''
        global_episode = self.global_episode.eval(sess)  # what is global_episode at runtime? -- variable with shape 0
        summary = sess.run(self.write_op,
                           feed_dict=feed_dict)

        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush()


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
        self.global_step = tf.Variable(
            0,
            trainable=False,
            name='global_step')

        self.global_episode = tf.Variable(
            0,
            trainable=False,
            name='global_episode')

        self.increment_global_episode = tf.assign(
            self.global_episode,
            self.global_episode + 1,
            name='increment_global_episode')

        self.score = tf.placeholder(tf.int32, [], name='score')
        self.screen_inputs = tf.placeholder(tf.int32, shape=[None, len(SCREEN_FEATURES), *self.screen_dims], name='screen_inputs')  # 17, 32, 32
        self.minimap_inputs = tf.placeholder(tf.int32, shape=[None, len(MINIMAP_FEATURES), *self.minimap_dims], name='minimap_inputs')
        self.flat_inputs = tf.placeholder(tf.float32, shape=[None, len(features.Player)], name='flat_inputs')  # env stuff: mins, gas, supply, etc -- 11 total

        self.inputs = tf.placeholder(tf.int32,[None, *self.screen_dims], name="inputs")  # [None, px1, px2]
        self.actions = tf.placeholder(tf.float32, [None, np.prod(self.screen_dims)], name='actions')  # e.g:
        self.targets = tf.placeholder(tf.float32, [None], name="targets")  #*** features.Player is 'broadcast vector'.

        # preprocessing per rays github
        self.screen_preprocessed = preprocess_spatial_features(self.screen_inputs, screen=True)
        self.minimap_preprocessed = preprocess_spatial_features(self.minimap_inputs, screen=False)
        self.flat_preprocessed = tf.log(self.flat_inputs + 1., name='flat_preprocessed')

        # Now onto screen and minimap. weight variables first
        screen_filters1 = tf.get_variable(name='screen_f1', shape=(5, 5, self.screen_preprocessed.shape[-1], 16))  # should be hwio
        screen_filters2 = tf.get_variable(name='screen_f2', shape=(3, 3, 16, 32))  # anything else for these?
        minimap_filters1 = tf.get_variable(name='minimap_f1', shape=(5, 5, self.minimap_preprocessed.shape[-1], 16))  # look back at ray's for this
        minimap_filters2 = tf.get_variable(name='minimap_f2', shape=(3, 3, 16, 32))

        # layers next
        screen_conv1 = tf.nn.conv2d(self.screen_preprocessed, screen_filters1, strides=[1, 1, 1, 1], padding='SAME', name='screen_conv1_in')
        screen_conv1 = tf.nn.relu(screen_conv1, name='screen_conv1_out')

        screen_conv2 = tf.nn.conv2d(screen_conv1, screen_filters2, strides=[1, 1, 1, 1], padding='SAME', name='screen_conv2_in')
        screen_conv2 = tf.nn.relu(screen_conv2, name='screen_conv2_out')

        minimap_conv1 = tf.nn.conv2d(self.minimap_preprocessed, minimap_filters1, strides=[1, 1, 1, 1], padding='SAME', name='minimap_conv1_in')
        minimap_conv1 = tf.nn.relu(minimap_conv1, name='minimap_conv1_out')

        minimap_conv2 = tf.nn.conv2d(minimap_conv1, minimap_filters2, strides=[1, 1, 1, 1], padding='SAME', name='minimap_conv2_in')
        minimap_conv2 = tf.nn.relu(minimap_conv2, name='minimap_conv2_out')

        # linear layer for non-spatial features (tanh activation)
        flat_linear = tf.layers.dense(self.flat_preprocessed, units=64, activation=tf.nn.tanh, name='flat_linear')  # i think this is 'info'/broadcast

        # flatten layers and concat  (would not flattening and concat along axis=3 work?) NO b/c flat_linear is [None, 64]
        screen_flat = tf.layers.flatten(screen_conv2, name='screen_flat')
        minimap_flat = tf.layers.flatten(minimap_conv2, name='minimap_flat')
        concat_layer = tf.concat([screen_flat, minimap_flat, flat_linear], axis=1, name='concat_layer')

        # state representation -- feel like not all this lines up with the paper how i would expect
        self.state_representation = tf.layers.dense(concat_layer, 256, activation=tf.nn.relu, name='state_rep')
        self.pred_q = tf.dense(  # diff between tf.dense and .fully_connected?
            inputs=self.state_representation,
            units=NUM_ACTIONS,
            name='q_pred')
        self.pred_action = tf.argmax(self.pred_q, axis=1)

        # spatial vs. non spatial starts down here
        # action function argument policies (nonspatial)
        # action function argument placeholders (for optimization)`
        self.argument_policy = dict()
        self.arguments = dict()

        for arg_type in actions.TYPES:
            # for spatial actions, represent each dimension independently
            if len(arg_type.sizes) > 1:
                if arg_type in SCREEN_TYPES:
                    units = self.screen_dims
                elif arg_type in MINIMAP_TYPES:
                    units = self.minimap_dims  # so these will be [n, n]

                arg_policy_x = tf.layers.dense(self.state_representation, units=units[0], activation=tf.nn.softmax)
                arg_policy_y = tf.layers.dense(self.state_representation, units=units[1], activation=tf.nn.softmax)

                self.argument_policy[str(arg_type) + 'x'] = arg_policy_x
                self.argument_policy[str(arg_type) + 'y'] = arg_policy_y

                arg_ph_x = tf.placeholder(tf.float32, [None, units[0]])
                arg_ph_y = tf.placeholder(tf.float32, [None, units[1]])

                self.arguments[str(arg_type) + 'x'] = arg_ph_x
                self.arguments[str(arg_type) + 'y'] = arg_ph_y

            else:
                arg_policy = tf.layers.dense(self.state_representation, units=arg_type.sizes[0], activation=tf.nn.softmax)
                self.argument_policy[str(arg_type)] = arg_policy

                arg_ph = tf.placeholder(tf.float32, [None, arg_type.sizes[0]])
                self.arguments[str(arg_type)] = arg_ph



    def _build_optimization(self):  # "params learnt with A3C" (using A2C here)
        ''' construct a graph for network updates '''
        # dont think i should need any of the probability stuff, just taking an arg max
        self.actions = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name='actions_ph')
        self.reward = tf.placeholder(tf.float32, shape=(None), name='reward_ph')
        self.target_q = tf.placeholder(tf.float32, [None], name='targets')  #*** features.Player is 'broadcast vector'.


        self.pred_q_action = tf.reduce_sum(self.pred_q * tf.one_hot(self.actions, depth=NUM_ACTIONS), axis=1)  # act_t_ph?
         # would this be better off in agent and ph here?
        self.target_q_action = self.reward + (1-self.done_mask) * self.gamma *tf.reduce_max(target_q, axis=1)
        self.loss = huber_loss(self.targets-self.prediction)

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
