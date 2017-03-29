import tensorflow as tf
import numpy as np

a1 = np.array([1, 2, 3])

v1 = tf.Variable(2.0, trainable=False)   # float32 scalar
v2 = tf.Variable([0.1], tf.float32)      # float32 (1,)

v3 = tf.Variable(tf.constant(0.1, shape=[2, 2]))   # float32 (2, 2)
v4 = tf.Variable(tf.constant(a1))                  # int64 (3,)

b1 = tf.Variable(tf.zeros([3]))                    # float32 (3,)
b2 = tf.Variable(tf.zeros([3], dtype=tf.int32))    # int32 (3,)
b3 = tf.Variable(tf.fill([3], 1))                  # int32 (3,)

W1 = tf.Variable(tf.truncated_normal([1], stddev=1.0))    # float32 (1,)
print(W1.get_shape())
print(W1)

W2 = tf.Variable(tf.truncated_normal([3, 2], stddev=1.0)) # float32 (3, 2)
W3 = tf.Variable(tf.truncated_normal([3, 2], mean=0, stddev=1.0, dtype=tf.float64))
W4 = tf.Variable(tf.random_uniform([1], -1, 1))        # float32 (1,)
W5 = tf.Variable(tf.random_uniform([1], -0.1, 0.1))    # float32 (1,)
W6 = tf.Variable(tf.random_uniform([2, 3], -0.1, 0.1)) # float32 (2, 3)
W7 = tf.Variable(tf.random_uniform([2, 3], -0.1, 0.1, dtype=tf.float64, name="W3"))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run([v1, v2, v3, v4]))
    print(sess.run([b1, b2, b3]))
    print(sess.run([W1, W2, W3, W4, W5, W6, W7]))


