from tensorflow.contrib import slim
from contextlib import ExitStack
import tensorflow as tf

def discriminator(net, layers, scope='adversary'):
        
    with ExitStack() as stack:
        stack.enter_context(tf.variable_scope(scope))
        stack.enter_context(
            slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(2.5e-5)))
        for dim in layers:
            net = slim.fully_connected(net, dim)
        net = slim.fully_connected(net, 2, activation_fn=None)
    return net
