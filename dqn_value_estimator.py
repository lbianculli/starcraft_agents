import numpy as np
import tensorflow as tf

class PlayerRelativeMovementCNN():
    ''' uses feature_screen.player_relative to assign q value to movements '''
    
    def __init__(self,
                 spatial_dims,
                 learning_rate,
                 save_path=None,
                 summ_path=None,
                 name='DQN'):
        ''' initialize instance-specific hyperparams, build tf graph '''
        
        self.spatial_dims = spatial_dims
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.summ_path = summ_path
        self.name = name
        
        self._build() # ...

    
    
    def _build(self):
        # this seems to be mostly the same, although doesnt follow paper specs exactly...
        # after conv, spatial output:
        self.output = tf.nn.conv2d(  # make sure this is right
            inputs=self.conv2,
            filter= weights['wfc'],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='output')
        
        self.flat = tf.layers.flatten(self.output, name='flat')  # ex: [None, 4, 4] --> [None, 16]
        
        # value estimate trackers for summaries
        self.max_q = tf.reduce_max(self.flat, name='max_q')
        self.mean_q = tf.reduce_mean(self.flat, name='mean_q')  # if axis=None, reduce along all dims (scalar)
        
        # optimization: MSE between state predicted Q and target Q
        self.prediction = tf.reduce_sum(
            tf.multiply(self.flat, self.actions)m
            axis=1,
            name='preds')
        
        self.loss = tf.reduce_mean(
            tf.square(self.targets - self.prediction),
            name='loss')
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, 
                                                                            global_step=self.global_step)  # what does this do?
        
        
        
  

