import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from cap_config import settings

from cap_datasets import mnist
import m_capsules

FLAGS = None


def preprocess_image(image):
  """ Scale the image value between -1 and 1.
    :param image: An image in Tensor.
    :return A scaled image in Tensor.
  """

  image = tf.to_float(image)
  image = tf.subtract(image, 128.0)
  image = tf.div(image, 128.0)
  return image


def load_batch(dataset, batch_size=32):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    # image: Tensor(28, 28, j1) label Tensor()
    image, label = data_provider.get(['image', 'label'])

    image = preprocess_image(image)

    # images: Tensor(?, 28, 28, 1) labels Tensor(?,)
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
    iterations_per_epoch = dataset.num_samples // FLAGS.batch_size # 60,000/24 = 2500

    # images: Tensor (?, 28, 28, 1)
    # labels: Tensor (?)
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size)

    # Tensor(?, 10)
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

    # poses: Tensor(?, 10, 4, 4) activations: (?, 10)
    poses, activations = m_capsules.nets.capsules_net(images, num_classes=10, iterations=3, batch_size=FLAGS.batch_size, name='capsules_em')

    global_step = tf.train.get_or_create_global_step()
    loss = m_capsules.nets.spread_loss(
        one_hot_labels, activations, iterations_per_epoch, global_step, name='spread_loss'
    )
    tf.summary.scalar('losses/spread_loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_tensor = slim.learning.create_train_op(
        loss, optimizer, global_step=global_step, clip_gradient_norm=4.0
    )

    slim.learning.train(
        train_tensor,
        logdir=FLAGS.log_dir,
        log_every_n_steps=10,
        save_summaries_secs=60,
        saver=tf.train.Saver(max_to_keep=2),
        save_interval_secs=600,
    )


if __name__ == '__main__':
    tf.app.run()

