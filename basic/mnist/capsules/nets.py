"""An implementation of matrix capsules with EM routing.
"""

import tensorflow as tf

from capsules.core import _conv2d_wrapper, capsules_init, capsules_conv, capsules_fc

slim = tf.contrib.slim

# ------------------------------------------------------------------------------#
# -------------------------------- capsules net --------------------------------#
# ------------------------------------------------------------------------------#

def capsules_net(inputs, num_classes, iterations, name='CapsuleEM-V0'):
  """Replicate the network in `Matrix Capsules with EM Routing.`
  """

  with tf.variable_scope(name) as scope:

    # [24, 28, 28, 1] -> conv 5x5 filters, strides 2, 32 channels -> [24, 14, 14, 32]
    nets = _conv2d_wrapper(
      inputs, shape=[5, 5, 1, 32], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu, name='conv1'
    )

    # [24, 14, 14, 32] -> conv2d, 1x1, strides 1, channels 32x(4x4+1)
    # -> (poses (24, 14, 14, 32, 4, 4), activations (24, 14, 14, 32))
    nets = capsules_init(nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], padding='VALID', pose_shape=[4, 4], name='capsule_init')

    # (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 2
    # -> (poses (24, 6, 6, 32, 4, 4), activations 24, 6, 6, 32))
    nets = capsules_conv(nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv1')

    # (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 1 -> (poses, activations)
    # -> (poses (24, 4, 4, 32, 4, 4), activations 24, 4, 4, 32))
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], iterations=iterations, name='capsule_conv2'
    )

    # (poses, activations) -> capsule-fc 1x1x32x10x4x4 shared view
    # -> (poses (24, 10, 4, 4), activations 24, 10))
    nets = capsules_fc(nets, num_classes, iterations=iterations, name='capsule_fc')

    poses, activations = nets

  return poses, activations

# ------------------------------------------------------------------------------#
# ------------------------------------ loss ------------------------------------#
# ------------------------------------------------------------------------------#

def spread_loss(labels, activations, iterations_per_epoch, global_step, name):
  """This adds spread loss to total loss.

  :param labels: [N, O], where O is number of output classes, one hot vector, tf.uint8.
  :param activations: [N, O], activations.
  :param margin: margin 0.2 - 0.9 fixed schedule during training.

  :return: spread loss
  """

  # margin schedule
  # margin increase from 0.2 to 0.9
  margin = tf.train.piecewise_constant(
    tf.cast(global_step, dtype=tf.int32),
    boundaries=[
      (iterations_per_epoch * x) for x in range(1, 8)
    ],
    values=[
      x / 10.0 for x in range(2, 10)
    ]
  )

  activations_shape = activations.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    mask_t = tf.equal(labels, 1)
    mask_i = tf.equal(labels, 0)

    activations_t = tf.reshape(
      tf.boolean_mask(activations, mask_t), [activations_shape[0], 1]
    )
    activations_i = tf.reshape(
      tf.boolean_mask(activations, mask_i), [activations_shape[0], activations_shape[1] - 1]
    )

    gap_mit = tf.reduce_sum(
      tf.square(
        tf.nn.relu(
          margin - (activations_t - activations_i)
        )
      )
    )

    # tf.add_to_collection(
    #   tf.GraphKeys.LOSSES, gap_mit
    # )
    #
    # total_loss = tf.add_n(
    #   tf.get_collection(
    #     tf.GraphKeys.LOSSES
    #   ), name='total_loss'
    # )

    tf.losses.add_loss(gap_mit)

    return gap_mit

# ------------------------------------------------------------------------------#

