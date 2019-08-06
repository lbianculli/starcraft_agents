""" 
i think I am going to plan on totally copying structure if I have too much trouble with params.
currently: batch_generator() returns everything that will be fed, and the SL file sets up the network and actual ops
Ray: network sets up phs and network. agent handles ops. Feed for params depends on spatial/non. Agent settles which to use.
"""
### from network
# ...
self.argument_policy = dict()
self.arguments = dict()
for arg_type in actions.TYPES:
  if len(arg_type.sizes) > 1:
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

-----------------------------------------------------
### from agent
# def get_batch(self, terminal) ...
        feed_dict = {self.network.screen_features: screen,
                     self.network.minimap_features: minimap,
                     self.network.flat_features: flat,
                     self.network.actions: actions,
                     self.network.returns: returns}  # essentially same, but dont have returns now

        # add args and arg_types to feed_dict
        net_args = self.network.arguments
        batch_size = len(arg_types)

        # first populate feed_dict with zero arrays
        for arg_type in sc2_actions.TYPES:
          if len(arg_type.sizes) > 1:
            if arg_type in SCREEN_TYPES:
                x_size = feature_screen_size[0]  # all these are 64
                y_size = feature_screen_size[1]
            elif arg_type in MINIMAP_TYPES:
                x_size = feature_minimap_size[0]
                y_size = feature_minimap_size[1]

            feed_dict[net_args[str(arg_type) + "x"]] = np.zeros(  # feed_dict["ph_name"]["arg_typex"]
                (batch_size, x_size))
            feed_dict[net_args[str(arg_type) + "y"]] = np.zeros(
                (batch_size, y_size))

            else:
              feed_dict[net_args[str(arg_type)]] = np.zeros(
                  (batch_size, arg_type.sizes[0]))

        # then one_hot encode args
        for step in range(batch_size):
          for i, arg_type in enumerate(arg_types[step]):
            if len(arg_type.sizes) > 1:
              arg_key_x = net_args[str(arg_type) + "x"]
              feed_dict[arg_key_x][step, args[step][i][0]] = 1

              arg_key_y = net_args[str(arg_type) + "x"]
              feed_dict[arg_key_y][step, args[step][i][1]] = 1
            else:
              arg_key = net_args[str(arg_type)]
              feed_dict[arg_key][step, args[step][i][0]] = 1
                    
        return feed_dict
      
      
def _train_network(self, terminal=False):
    feed_dict = self._get_batch(terminal)
    self.network.optimizer_op(self.sess, feed_dict)
    return feed_dict
