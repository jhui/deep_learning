import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

def variable_summaries(var):
  """Attach mean/max/min/sd & histogram for TensorBoard visualization."""
  with tf.name_scope('summaries'):
    # Find the mean of the variable say W.
    mean = tf.reduce_mean(var)
    # Log the mean as scalar
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    # Log var as a histogram
    tf.summary.histogram('histogram', var)

def affine_layer(x, name, shape, keep_prob, act_fn=tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0 / shape[0])))
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([shape[1]]))
            variable_summaries(b)
        with tf.name_scope('z'):
            z = tf.matmul(x, W) + b
            tf.summary.histogram('summaries/histogram', z)

        h = act_fn(tf.matmul(x, W) + b)
        with tf.name_scope('out/summaries'):
            tf.summary.histogram('histogram', h)

        with tf.name_scope('dropout/summaries'):
            dropped = tf.nn.dropout(h, keep_prob)
            tf.summary.histogram('histogram', dropped)

        return dropped, W

def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.summary.scalar('dropoout_probability', keep_prob)

  x = tf.placeholder(tf.float32, [None, 784])

  image = tf.reshape(x[:1], [-1, 28, 28, 1])
  tf.summary.image("image", image)

  h1, _ = affine_layer(x, 'layer1', [784, 256], keep_prob)
  h2, _ = affine_layer(h1, 'layer2', [256, 100], keep_prob)
  y, W3 = affine_layer(h2, 'output', [100, 10], keep_prob=1.0, act_fn=tf.identity)

  labels = tf.placeholder(tf.float32, [None, 10])

  lmbda = tf.placeholder(tf.float32)
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y) +
         lmbda * (tf.nn.l2_loss(W3)))

  tf.summary.scalar('loss', cross_entropy)

  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

  summary = tf.summary.merge_all()

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
      sess.run(init)
      for step in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5, keep_prob:0.5})
        if step % 20 == 0:
          # Update the events file.
          summary_str = sess.run(summary, feed_dict={x: batch_xs, labels: batch_ys, lmbda:5e-5, keep_prob:0.5})
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          labels: mnist.test.labels, keep_prob:0.5}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/log',
                      help='Directory for log')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# 0.9816
