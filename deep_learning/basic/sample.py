from gan_env import *
from glob import glob
import numpy as np
import tensorflow as tf

c1 = tf.placeholder(tf.float32, [2, 2, 2, 3])
c2 = tf.placeholder(tf.float32, [2, 3])

shape = tf.shape(c1)
v2 = tf.tile(c2, [1, shape[1]*shape[2]])
v2 = tf.reshape(v2, [shape[0], shape[1], shape[2], shape[3]])
v3 = tf.concat([c1, v2], axis=3)

with tf.Session() as sess:
    d1 = np.array([ [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], [[[12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 24]]] ])

    d2 = np.array([[24, 25, 26], [27, 28, 29]])
    v3 = sess.run([v3], feed_dict={c1:d1, c2:d2})
    print(v3)
