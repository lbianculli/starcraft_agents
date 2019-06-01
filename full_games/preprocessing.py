"""Tensorflow graphs that preprocess pysc2 features."""
import tensorflow as tf

from pysc2.lib import features as sc2_features



SCREEN_FEATURES = [(feature.type, feature.scale) for feature in sc2_features.SCREEN_FEATURES]  # i believe scale is a pixel-valued int
MINIMAP_FEATURES = [(feature.type, feature.scale)for feature in sc2_features.MINIMAP_FEATURES]
# so these will be e.g: (FeatureType.SCALAR, 256) or (FeatureType.CATEGORICAL, 5)


def preprocess_spatial_features(features, screen=True):
    ''' preprocessing per deepmind papaer. features will be placeholder of [None, len(features)]'''
    if screen:
        feature_specs = SCREEN_FEATURES
    else:
        feature_specs = MINIMAP_FEATURES

    # transpose from (batch, channels, y, x) to (batch, x, y, channels)
    transposed = tf.transpose(features, perm=[0, 3, 2, 1], name='transposed')
    # transposed = tf.transpose(features, perm=[0, 2, 1], name='transposed')


    preprocess_ops = []
    for idx, (feature_type, scale) in enumerate(feature_specs):  # looks like scale would be the int value from examples up top
        layer = transposed[:, :, :, idx]  # x, y, c, idx
        if feature_type == sc2_features.FeatureType.CATEGORICAL:
            # one-hot encode in channel dimension -> 1x1 convolution
            one_hot = tf.one_hot(layer, depth=scale, axis=-1, name='preprocess_categorical')
            # embed_filter = tf.Variable(expected_shape=[1, 1, scale, 1],)  # how do i do this? it will be a variable but how to do shape or name?
            # embed = tf.nn.conv2d(one_hot, filter=embed_filter, strides=[1, 1, 1, 1], padding='SAME')  # filter: [h, w, c_in, c_out]
            embed = tf.layers.conv2d(inputs=one_hot, filters=1, kernel_size=[1, 1], strides=[1, 1], padding='SAME')  # workaround for now
            # would like to figure out the above though, have had similar problems before

            preprocess_ops.append(embed)

        elif feature_type == sc2_features.FeatureType.SCALAR:
            # if not categorical, log norm the value, make tensor
            transform = tf.log(tf.cast(layer, tf.float32) + 1., name='preprocess_scalar')
            preprocess_ops.append(tf.expand_dims(transform, -1))

    preprocessed = tf.concat(preprocess_ops, -1)
    return preprocessed


