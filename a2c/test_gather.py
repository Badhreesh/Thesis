import tensorflow as tf
import numpy as np

a = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

i = [0, 1, 2]

one_hot = tf.one_hot(i, 3)

test = tf.reduce_sum(tf.multiply(a, one_hot),1)

sess= tf.Session()
print(a.get_shape())
#print(sess.run(test))
