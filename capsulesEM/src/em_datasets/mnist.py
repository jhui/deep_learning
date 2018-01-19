"""Functions for downloading and loading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

from . import utils

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TESTS_IMAGES = 't10k-images-idx3-ubyte.gz'
TESTS_LABELS = 't10k-labels-idx1-ubyte.gz'

NUM_TRAIN_EXAMPLES = 60000
NUM_TESTS_EXAMPLES = 10000


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_images(f):
  """Extract the mnist images into 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    images: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: if the bytestream doesn't start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
        'Invalid magic number {:d} in MNIST image file: {:s}'.format(
          magic, f.name
        )
      )
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    _buffer = bytestream.read(rows * cols * num_images)
    images = np.frombuffer(_buffer, dtype=np.uint8)
    images = images.reshape(num_images, rows, cols, 1)
    return images


def _extract_labels(f):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
        'Invalid magic number {:d} in MNIST label file: {:s}'.format(
          magic, f.name
        )
      )
    num_labels = _read32(bytestream)
    _buffer = bytestream.read(num_labels)
    labels = np.frombuffer(_buffer, dtype=np.uint8)
    return labels


def _dense_to_one_hot(labels, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros(
    shape=(num_labels, num_classes), dtype=np.uint8
  )
  labels_one_hot.flat[index_offset + labels.ravel()] = 1
  return labels_one_hot


def load_mnist(
  data_directory,
  is_training=True,
  normalize=True,
  one_hot=True,
  num_classes=10
):
  """Load mnist train/tests images and labels into numpy array.

  Args:
    data_directory: directory contains train/tests images and labels.
    normalize: if True, images are normalized by pixel / 255.0 - 0.5.

  Return:
    images, labels: images in 4D numpy array [index, y, x, depth],
      and labels in 1D numpy array [index] if one_hot is False,
      and labels in 2D numpy array [index, offset] if one_hot is True.

  """
  # train vs tests
  if is_training:
    _images_filename = TRAIN_IMAGES
    _labels_filename = TRAIN_LABELS
  else:
    _images_filename = TESTS_IMAGES
    _labels_filename = TESTS_LABELS

  # load images
  _images_filepath = utils.download(
    _images_filename, data_directory, DEFAULT_SOURCE_URL, overwrite=False
  )

  with gfile.Open(_images_filepath, 'rb') as f:
    images = _extract_images(f)

  if normalize:
    images = images.astype(np.float32) / 255.0 - 0.5

  # load labels
  _labels_filepath = utils.download(
    _labels_filename, data_directory, DEFAULT_SOURCE_URL, overwrite=False
  )

  with gfile.Open(_labels_filepath, 'rb') as f:
    labels = _extract_labels(f)

  if one_hot:
    labels = _dense_to_one_hot(labels, num_classes)

  return images, labels


def inputs(data_directory, is_training, batch_size):
  """This constructs batched inputs of mnist data.
  """

  images, labels = load_mnist(
    data_directory=data_directory,
    is_training=is_training
  )

  data_queues = tf.train.slice_input_producer([images, labels])

  images, labels = tf.train.shuffle_batch(
    data_queues,
    num_threads=16,
    batch_size=batch_size,
    capacity=batch_size * 64,
    min_after_dequeue=batch_size * 32,
    allow_smaller_final_batch=False
  )

  # images: Tensor (24, 28, 28, 1), labels: Tensor (24, 10)
  return (images, labels)

