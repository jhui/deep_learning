"""Trains and Evaluates the MNIST network using a feed dictionary."""

import tensorflow as tf
import sys
import time
import math

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

HIDDEN1, HIDDEN2 = 128, 32

LEARNING_RATE = 0.01
ITERATIONS = 2000
BATCH_SIZE = 100

INPUT_DATA_DIR = '/tmp/tensorflow/mnist/input_data'

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def get_next_batch(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):

  true_count = 0 
  steps_per_epoch = data_set.num_examples // BATCH_SIZE
  num_examples = steps_per_epoch * BATCH_SIZE
  for step in range(steps_per_epoch):
    feed_dict = get_next_batch(data_set, images_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  """Train MNIST for a number of steps."""
  data_sets = read_data_sets(INPUT_DATA_DIR)

  with tf.Graph().as_default():
    images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

    # Hidden 1.
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, HIDDEN1],
                                stddev=1.0 / math.sqrt(IMAGE_PIXELS)),
            name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN1]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([HIDDEN1, HIDDEN2],
                                stddev=1.0 / math.sqrt(HIDDEN1)),
            name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN2]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([HIDDEN2, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(HIDDEN2)),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    # Add to the Graph the Ops for loss calculation.
    labels = tf.to_int64(labels_placeholder)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    loss =  tf.reduce_mean(cross_entropy, name='xentropy_mean')

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    correct = tf.nn.in_top_k(logits, labels, 1)
    eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

    init = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init)

    for step in range(ITERATIONS):
      start_time = time.time()

      feed_dict = get_next_batch(data_sets.train, images_placeholder, labels_placeholder)

      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

      if (step + 1) % 1000 == 0 or (step + 1) == ITERATIONS:
        print('Training Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
        print('Validation Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
        print('Test Data Eval:')
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)


def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])


