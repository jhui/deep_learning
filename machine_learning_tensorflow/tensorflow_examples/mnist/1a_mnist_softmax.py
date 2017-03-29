
"""A very simple MNIST classifier.
"""

import argparse
import sys

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import tensorflow as tf

def main(_):
  # Import data
  mnist = read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  trainer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(trainer, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
