import tensorflow as tf
import numpy as np
from baselines.a2c.utils import conv, fc, conv_to_fc, ortho_init # ortho_init used by conv
from adda_doom import register_model_fn

@register_model_fn('nature_cnn')
def nature_cnn(unscaled_images, scope, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    #unscaled_images = tf.placeholder(tf.float32, shape=[None, 84, 84, 1], name='unscaled_images')
    with tf.variable_scope(scope):
        scaled_images = tf.cast(unscaled_images, tf.float32) / 255. 
        activ = tf.nn.relu
        # 8x8 filter size is common on the very 1st conv layer, looking at the input image
        h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
        h3 = conv_to_fc(h3)
        return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

@register_model_fn('nature_cnn_no_fc')
def nature_cnn_no_fc(unscaled_images, scope, **conv_kwargs):
    """
    CNN from Nature paper without FC Layer
    """
    #unscaled_images = tf.placeholder(tf.float32, shape=[None, 84, 84, 1], name='unscaled_images')
    with tf.variable_scope(scope):
        scaled_images = tf.cast(unscaled_images, tf.float32) / 255. 
        activ = tf.nn.relu
        # 8x8 filter size is common on the very 1st conv layer, looking at the input image
        h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
        h3 = conv_to_fc(h3)
        return h3
