import tensorflow as tf
from datasets import mnist

import capsules
import numpy as np
from config import settings

import tensorflow.contrib.slim as slim

FLAGS = None

def lenet(images):
    net = slim.conv2d(images, 20, [5,5], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 50, [5,5], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 500, scope='fc4')
    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
    return net

def preprocess_image(image):
  """Preprocesses the given image.
  Args:
    image: A `Tensor` representing an image of arbitrary size.
  Returns:
    A preprocessed image.
  """
  image = tf.to_float(image)
  image = tf.subtract(image, 128.0)
  image = tf.div(image, 128.0)
  return image

def load_batch(dataset, batch_size=32, height=28, width=28, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    image, label = data_provider.get(['image', 'label'])

    image = preprocess_image(
        image,
        height,
        width,
        is_training)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, labels

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS = settings()

    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    # Slim dataset contains data sources, decoder, reader and other meta-information
    dataset = mnist.get_split('train', FLAGS.dataset_dir)

    # images: Tensor (?, 28, 28, 1)
    # labels: Tensor (?)
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)

    iterations_per_epoch = dataset.num_samples // FLAGS.batch_size # 60,000/24 = 2500

    global_step = tf.train.get_or_create_global_step()

    poses, activations = capsules.nets.capsules_net(images, num_classes=10, iterations=3, name='capsulesEM-V0')

    loss = capsules.nets.spread_loss(
        labels, activations, iterations_per_epoch, global_step, name='spread_loss'
    )

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    tf.losses.softmax_cross_entropy(one_hot_labels, predictions)

    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    # use RMSProp to optimize
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        global_step=global_step,
        summarize_gradients=True)

    # run training
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        save_summaries_secs=20)


if __name__ == '__main__':
    tf.app.run()

