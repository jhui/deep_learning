import argparse
import sys
import tempfile
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.variable_scope('conv1'):
    h_conv1 = tf.layers.conv2d(
      inputs=x_image,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling layer - downsamples by 2X.
  with tf.variable_scope('pool1'):
    h_pool1 = tf.layers.max_pooling2d(inputs=h_conv1, pool_size=[2, 2], strides=2)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.variable_scope('conv2'):
    h_conv2 = tf.layers.conv2d(
      inputs=h_pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling layer with 2nd convolutional layer
  with tf.variable_scope('pool2'):
    h_pool2 = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.variable_scope('fc1'):
    keep_prob = tf.placeholder(tf.float32)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.layers.dense(inputs=h_pool2_flat, units=1024, activation=tf.nn.relu)
    h_fc1_drop = tf.layers.dropout(
      inputs=h_fc1, rate=keep_prob, training=keep_prob<1.0)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.variable_scope('fc2'):
    y_conv = tf.layers.dense(inputs=h_fc1_drop, units=10)
  return y_conv, keep_prob


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model

  x = tf.placeholder(tf.float32, [None, 784])

  # placeholder for true label
  y_ = tf.placeholder(tf.int64, [None])

  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):

      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
