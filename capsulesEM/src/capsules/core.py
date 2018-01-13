"""An implementation of matrix capsules with EM routing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import initializers

import tensorflow as tf

slim = tf.contrib.slim

epsilon = 1e-9

# ------------------------------------------------------------------------------#
# ------------------------------------ init ------------------------------------#
# ------------------------------------------------------------------------------#

def _matmul_broadcast(x, y, name):
  """Compute x @ y, broadcasting over the first `N - 2` ranks.
  """
  with tf.variable_scope(name) as scope:
    return tf.reduce_sum(
      x[..., tf.newaxis] * y[..., tf.newaxis, :, :], axis=-2
    )


def _get_variable_wrapper(
  name, shape=None, dtype=None, initializer=None,
  regularizer=None,
  trainable=True,
  collections=None,
  caching_device=None,
  partitioner=None,
  validate_shape=True,
  custom_getter=None
):
  """Wrapper over tf.get_variable().
  """

  with tf.device('/cpu:0'):
    var = tf.get_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, trainable=trainable,
      collections=collections, caching_device=caching_device,
      partitioner=partitioner, validate_shape=validate_shape,
      custom_getter=custom_getter
    )
  return var


def _get_weights_wrapper(
  name, shape, dtype=tf.float32, initializer=initializers.xavier_initializer(),
  weights_decay_factor=None
):
  """Wrapper over _get_variable_wrapper() to get weights, with weights decay factor in loss.
  """

  weights = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )

  if weights_decay_factor is not None and weights_decay_factor > 0.0:

    weights_wd = tf.multiply(
      tf.nn.l2_loss(weights), weights_decay_factor, name=name + '/l2loss'
    )

    tf.add_to_collection('losses', weights_wd)

  return weights


def _get_biases_wrapper(
  name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0)
):
  """Wrapper over _get_variable_wrapper() to get bias.
  """

  biases = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )

  return biases

# ------------------------------------------------------------------------------#
# ------------------------------------ main ------------------------------------#
# ------------------------------------------------------------------------------#

def _conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.conv2d().
  """

  with tf.variable_scope(name) as scope:
    kernel = _get_weights_wrapper(
      name='weights', shape=shape, weights_decay_factor=0.0
    )
    output = tf.nn.conv2d(
      inputs, filter=kernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _separable_conv2d_wrapper(inputs, depthwise_shape, pointwise_shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.separable_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=depthwise_shape, weights_decay_factor=0.0
    )
    pkernel = _get_weights_wrapper(
      name='pointwise_weights', shape=pointwise_shape, weights_decay_factor=0.0
    )
    output = tf.nn.separable_conv2d(
      input=inputs, depthwise_filter=dkernel, pointwise_filter=pkernel,
      strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[pointwise_shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _depthwise_conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.depthwise_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=shape, weights_decay_factor=0.0
    )
    output = tf.nn.depthwise_conv2d(
      inputs, filter=dkernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      d_ = output.get_shape()[-1].value
      biases = _get_biases_wrapper(
        name='biases', shape=[d_]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output

# ------------------------------------------------------------------------------#
# ---------------------------------- capsules ----------------------------------#
# ------------------------------------------------------------------------------#

def capsules_init(inputs, shape, strides, padding, pose_shape, name):
  """This constructs a primary capsule layer from a regular convolution layer.

  :param inputs: a regular convolution layer with shape [N, H, W, C],
    where often N is batch_size, H is height, W is width, and C is channel.
  :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
    where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
  :param strides: strides [1, SH, SW, 1] w.r.t [N, H, W, C], often [1, 1, 1, 1], or [1, 2, 2, 1].
  :param padding: padding, often SAME or VALID.
  :param pose_shape: the shape of each pose matrix, [PH, PW],
    where PH is pose height, and PW is pose width.
  :param name: name.

  :return: (poses, activations),
    poses: [N, H, W, C, PH, PW], activations: [N, H, W, C],
    where often N is batch_size, H is output height, W is output width, C is output channels,
    and PH is pose height, and PW is pose width.

  note: with respect to the paper, matrix capsules with EM routing, figure 1,
    this function provides the operation to build from
    ReLU Conv1 [batch_size, 14, 14, A] to
    PrimaryCapsule poses [batch_size, 14, 14, B, 4, 4], activations [batch_size, 14, 14, B] with
    Kernel [A, B, 4 x 4 + 1], specifically,
    weight kernel shape [1, 1, A, B], strides [1, 1, 1, 1], pose_shape [4, 4]
  """

  # assert len(pose_shape) == 2

  with tf.variable_scope(name) as scope:

    # poses: build one by one
    # poses = []
    # for ph in range(pose_shape[0]):
    #   poses_wire = []
    #   for pw in range(pose_shape[1]):
    #     poses_unit = _conv2d_wrapper(
    #       inputs, shape=shape, strides=strides, padding=padding, add_bias=False, activation_fn=None, name=name+'_pose_'+str(ph)+'_'+str(pw)
    #     )
    #     poses_wire.append(poses_unit)
    #   poses.append(tf.stack(poses_wire, axis=-1, name=name+'_poses_'+str(ph)))
    # poses = tf.stack(poses, axis=-1, name=name+'_poses')

    # poses: simplified build all at once
    poses = _conv2d_wrapper(
      inputs,
      shape=shape[0:-1] + [shape[-1] * pose_shape[0] * pose_shape[1]],
      strides=strides,
      padding=padding,
      add_bias=False,
      activation_fn=None,
      name='pose_stacked'
    )
    # poses = slim.conv2d(
    #   inputs,
    #   num_outputs=shape[-1] * pose_shape[0] * pose_shape[1],
    #   kernel_size=shape[0:2],
    #   stride=strides[1],
    #   padding=padding,
    #   activation_fn=None,
    #   weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04),
    #   scope='poses_stacked'
    # )
    # shape: poses_shape[0:-1] + [shape[-1], pose_shape[0], pose_shape[1]]
    # modified to [-1] + poses_shape[1:-1] + [shape[-1], pose_shape[0], pose_shape[1]]
    # to allow_smaller_final_batch dynamic batch size
    poses_shape = poses.get_shape().as_list()
    poses = tf.reshape(
      poses, shape=[-1] + poses_shape[1:-1] + [shape[-1], pose_shape[0], pose_shape[1]], name='poses'
    )

    activations = _conv2d_wrapper(
      inputs,
      shape=shape,
      strides=strides,
      padding=padding,
      add_bias=False,
      activation_fn=tf.sigmoid,
      name='activation'
    )
    # activations = slim.conv2d(
    #   inputs,
    #   num_outputs=shape[-1],
    #   kernel_size=shape[0:2],
    #   stride=strides[1],
    #   padding=padding,
    #   activation_fn=tf.sigmoid,
    #   weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04),
    #   scope='activations'
    # )
    # activations = tf.Print(
    #   activations, [activations.shape, activations[0, 4:7, 5:8, :]], str(activations.name) + ':', summarize=20
    # )

    # add into GraphKeys.SUMMARIES
    tf.summary.histogram(
      'activations', activations
    )

  return poses, activations


def capsules_conv(inputs, shape, strides, iterations, name):
  """This constructs a convolution capsule layer from a primary or convolution capsule layer.
      i: input capsules (32)
      o: output capsules (32)
      batch size: 24
      spatial dimension: 14x14
      kernel: 3x3
  :param inputs: a primary or convolution capsule layer with poses and activations
         pose: (24, 14, 14, 32, 4, 4)
         activation: (24, 14, 14, 32)
  :param shape: the shape of convolution operation kernel, [kh, kw, i, o] = (3, 3, 32, 32)
  :param strides: often [1, 1, 1, 1], or [1, 2, 2, 1].
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (poses, activations) same as capsule_init().

  """
  # Extract the poses and activations
  inputs_poses, inputs_activations = inputs
  inputs_poses_shape = inputs_poses.get_shape().as_list()

  assert shape[2] == inputs_poses_shape[3]
  assert strides[0] == strides[-1] == 1

  with tf.variable_scope(name) as scope:

    # kernel: (kh, kw, i, o, 4, 4) = (3, 3, 32, 32, 4, 4)
    kernel = _get_weights_wrapper(
      name='pose_view_transform_weights', shape=shape + [inputs_poses_shape[-1], inputs_poses_shape[-1]]
    )

    # Prepare inputs_poses_patches such that it can multiply it with kernel (aka broadcasting)
    # inputs_poses -> input_poses_patches
    # (24, 14, 14, 32, 4, 4) -> (24, 6,6, 3, 3, 32, 32, 4, 4)
    hk_offsets = [
      [(h_offset + k_offset) for k_offset in range(0, shape[0])] for h_offset in
      range(0, inputs_poses_shape[1] + 1 - shape[0], strides[1])
    ]
    wk_offsets = [
      [(w_offset + k_offset) for k_offset in range(0, shape[1])] for w_offset in
      range(0, inputs_poses_shape[2] + 1 - shape[1], strides[2])
    ]
    inputs_poses_patches = tf.transpose(
      tf.gather(
        tf.gather(
          inputs_poses, hk_offsets, axis=1, name='gather_poses_height_kernel'
        ), wk_offsets, axis=3, name='gather_poses_width_kernel'
      ), perm=[0, 1, 3, 2, 4, 5, 6, 7], name='inputs_poses_patches'
    )
    inputs_poses_patches = inputs_poses_patches[..., tf.newaxis, :, :]
    inputs_poses_patches = tf.tile(
      inputs_poses_patches, [1, 1, 1, 1, 1, 1, shape[-1], 1, 1], name='workaround_broadcasting_issue'
    )

    # Compute the vote
    # (24, 6, 6, 3, 3, 32, 32, 4, 4)
    votes = _matmul_broadcast(
      inputs_poses_patches, kernel, name='inputs_poses_patches_view_transformation'
    )

    # votes -> (24, 6, 6, 3x3x32=288, 32, 16)
    votes_shape = votes.get_shape().as_list()
    votes = tf.reshape(
      votes, [
        -1,  votes_shape[1],  votes_shape[2],
        votes_shape[3] * votes_shape[4] * votes_shape[5],
        votes_shape[6],  votes_shape[7] * votes_shape[8]
      ], name='votes'
    )

    # (24, 6, 6, 3, 3, 32)
    inputs_activations_patches = tf.transpose(
      tf.gather(
        tf.gather(
          inputs_activations, hk_offsets, axis=1, name='gather_activations_height_kernel'
        ), wk_offsets, axis=3, name='gather_activations_width_kernel'
      ), perm=[0, 1, 3, 2, 4, 5], name='inputs_activations_patches'
    )

    # (24, 6, 6, 288)
    i_activations = tf.reshape(
      inputs_activations_patches, [
        -1,  votes_shape[1],  votes_shape[2],
        votes_shape[3] * votes_shape[4] * votes_shape[5]
      ], name='i_activations'
    )

    # beta_v and beta_a one for each output capsule: (1, 1, 1, 32)
    beta_v = _get_weights_wrapper(
      name='beta_v', shape=[1, 1, 1, votes_shape[6]]
    )
    beta_a = _get_weights_wrapper(
      name='beta_a', shape=[1, 1, 1, votes_shape[6]]
    )

    # Use EM routing to compute the pose and activation
    # poses (24, 6, 6, 32, 16)
    # activation (24, 6, 6, 32)
    poses, activations = matrix_capsules_em_routing(
      votes, i_activations, beta_v, beta_a, iterations, name='em_routing'
    )

    # (24, 6, 6, 32, 4, 4)
    poses = tf.reshape(
      poses, [
        -1, votes_shape[1], votes_shape[2], votes_shape[6], votes_shape[7], votes_shape[8]
      ]
    )

  return poses, activations


def capsules_fc(inputs, num_classes, iterations, name):
  """This constructs an output layer from a primary or convolution capsule layer via
    a full-connected operation with one view transformation kernel matrix shared across each channel.

  :param inputs: a primary or convolution capsule layer with poses and activations,
    poses shape [N, H, W, C, PH, PW], activations shape [N, H, W, C]
  :param num_classes: number of classes.
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (pose, activation) same as capsule_init().

  note: with respect to the paper, matrix capsules with EM routing, figure 1,
    This is the D -> E in figure.
    This step includes two major sub-steps:
      1. Apply one view transform weight matrix PH x PW (4 x 4) to each input channel, this view transform matrix is
        shared across (height, width) locations. This is the reason the kernel labelled in D has 1 x 1, and the reason
        the number of variables of weights is D x E x 4 x 4.
      2. Re-struct the inputs vote from [N, H, W, I, PH, PW] into [N, H x W x I, PH x PW],
        add scaled coordinate on first two elements, EM routing an output [N, O, PH x PW],
        and reshape output [N, O, PH, PW].
    The difference between fully-connected layer and convolution layer, is that:
      1. The corresponding kernel size KH, KW in this fully-connected here is actually the whole H, W, instead of 1, 1.
      2. The view transformation matrix is shared within KH, KW (i.e., H, W) in this fully-connected layer,
        whereas in the convolution capsule layer, the view transformation can be different for each capsule
        in the KH, KW, but shared across different (height, width) locations.
  """

  inputs_poses, inputs_activations = inputs

  inputs_poses_shape = inputs_poses.get_shape().as_list()

  inputs_activations_shape = inputs_activations.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    # kernel: [I, O, PW, PW]
    # yg note: if pose is irregular such as 5x3, then kernel for pose view transformation should be 3x3.
    kernel = _get_weights_wrapper(
      name='pose_view_transform_weights',
      shape=[
        inputs_poses_shape[3], num_classes, inputs_poses_shape[-1], inputs_poses_shape[-1]
      ],
    )

    # inputs_pose_expansion: [N, H, W, I, 1, PH, PW]
    # inputs_pose_expansion: expand inputs_pose dimension to match with kernel for broadcasting,
    # share the transformation matrices between different positions of the same capsule type,
    # share the transformation matrices as kernel (1, 1) broadcasting to inputs pose expansion (H, W)
    inputs_poses_expansion = inputs_poses[..., tf.newaxis, :, :]

    # TODO: update when issue fixed for broadcasting: https://github.com/tensorflow/tensorflow/issues/14924
    # temporary workaround with tf.tile.
    inputs_poses_expansion = tf.tile(
      inputs_poses_expansion, [1, 1, 1, 1, num_classes, 1, 1], name='workaround_broadcasting_issue'
    )

    # votes: [N, H, W, I, O, PH, PW]
    votes = _matmul_broadcast(
      inputs_poses_expansion, kernel, name='votes'
    )
    votes_shape = votes.get_shape().as_list()
    # votes: reshape into [N, H, W, I, O, PH x PW]
    votes = tf.reshape(
      votes, [-1] + votes_shape[1:-2] + [votes_shape[-2] * votes_shape[-1]]
    )
    # stop gradient on votes
    # votes = tf.stop_gradient(votes, name='votes_stop_gradient')

    # add scaled coordinate (row, column) of the center of the receptive field of each capsule
    # to the first two elements of its vote
    H = inputs_poses_shape[1]
    W = inputs_poses_shape[2]

    coordinate_offset_hh = tf.reshape(
      (tf.range(H, dtype=tf.float32) + 0.50) / H, [1, H, 1, 1, 1]
    )
    coordinate_offset_h0 = tf.constant(
      0.0, shape=[1, H, 1, 1, 1], dtype=tf.float32
    )
    coordinate_offset_h = tf.stack(
      [coordinate_offset_hh, coordinate_offset_h0] + [coordinate_offset_h0 for _ in range(14)], axis=-1
    )

    coordinate_offset_ww = tf.reshape(
      (tf.range(W, dtype=tf.float32) + 0.50) / W, [1, 1, W, 1, 1]
    )
    coordinate_offset_w0 = tf.constant(
      0.0, shape=[1, 1, W, 1, 1], dtype=tf.float32
    )
    coordinate_offset_w = tf.stack(
      [coordinate_offset_w0, coordinate_offset_ww] + [coordinate_offset_w0 for _ in range(14)], axis=-1
    )

    votes = votes + coordinate_offset_h + coordinate_offset_w

    # votes: reshape into [N, H x W x I, O, PH x PW]
    # votes = tf.reshape(
    #   votes, [
    #     votes_shape[0],
    #     votes_shape[1] * votes_shape[2] * votes_shape[3],
    #     votes_shape[4],  votes_shape[5] * votes_shape[6]
    #   ]
    # )
    votes = tf.reshape(
      votes, [
        -1,
        votes_shape[1] * votes_shape[2] * votes_shape[3],
        votes_shape[4],  votes_shape[5] * votes_shape[6]
      ]
    )

    # inputs_activations: [N, H, W, I]
    # inputs_activations: reshape into [N, H x W x I]
    # i_activations = tf.reshape(
    #   inputs_activations, [
    #     inputs_activations_shape[0],
    #     inputs_activations_shape[1] * inputs_activations_shape[2] * inputs_activations_shape[3]
    #   ]
    # )
    i_activations = tf.reshape(
      inputs_activations, [
        -1,
        inputs_activations_shape[1] * inputs_activations_shape[2] * inputs_activations_shape[3]
      ]
    )

    # beta_v and beta_a one for each output capsule: [1, O]
    beta_v = _get_weights_wrapper(
      name='beta_v', shape=[1, num_classes]
    )
    beta_a = _get_weights_wrapper(
      name='beta_a', shape=[1, num_classes]
    )

    # output poses and activations via matrix capsules_em_routing algorithm
    # poses: [N, O, PH x PW], activations: [N, O]
    poses, activations = matrix_capsules_em_routing(
      votes, i_activations, beta_v, beta_a, iterations, name='em_routing'
    )

    # pose: [N, O, PH, PW]
    # poses = tf.reshape(
    #   poses, [
    #     votes_shape[0], votes_shape[4], votes_shape[5], votes_shape[6]
    #   ]
    # )
    poses = tf.reshape(
      poses, [
        -1, votes_shape[4], votes_shape[5], votes_shape[6]
      ]
    )

    # add into GraphKeys.SUMMARIES
    tf.summary.histogram(
      'activations', activations
    )

  return poses, activations


def matrix_capsules_em_routing(votes, i_activations, beta_v, beta_a, iterations, name):
  """The EM routing between input capsules (i) and output capsules (j).

  :param votes: (N, OH, OW, kh x kw x i, o, 4 x 4) = (24, 6, 6, 3x3*32=288, 32, 16)
  :param i_activation: activation from Level L (24, 6, 6, 288)
  :param beta_v: (1, 1, 1, 32)
  :param beta_a: (1, 1, 1, 32)
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.

  :return: (pose, activation) of output capsules.
  """

  votes_shape = votes.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    # note: match rr shape, i_activations shape with votes shape for broadcasting in EM routing

    # rr: [3x3x32=288, 32, 1]
    # rr: routing matrix from each input capsule (i) to each output capsule (o)
    rr = tf.constant(
      1.0/votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32
    )

    # i_activations: expand_dims to (24, 6, 6, 288, 1, 1)
    i_activations = i_activations[..., tf.newaxis, tf.newaxis]

    # beta_v and beta_a: expand_dims to (1, 1, 1, 1, 32, 1]
    beta_v = beta_v[..., tf.newaxis, :, tf.newaxis]
    beta_a = beta_a[..., tf.newaxis, :, tf.newaxis]

    def m_step(rr, votes, i_activations, beta_v, beta_a, inverse_temperature):
      """The M-Step in EM Routing from input capsules i to output capsule j.
      i: input capsules (32)
      o: output capsules (32)
      h: 4x4 = 16
      output spatial dimension: 6x6
      :param rr: routing assignments. shape = (kh x kw x i, o, 1) =(3x3x32, 32, 1) = (288, 32, 1)
      :param votes. shape = (N, OH, OW, kh x kw x i, o, 4x4) = (24, 6, 6, 288, 32, 16)
      :param i_activations: input capsule activation (at Level L). (N, OH, OW, kh x kw x i, 1, 1) = (24, 6, 6, 288, 1, 1)
         with dimensions expanded to match votes for broadcasting.
      :param beta_v: Trainable parameters in computing cost (1, 1, 1, 1, 32, 1)
      :param beta_a: Trainable parameters in computing next level activation (1, 1, 1, 1, 32, 1)
      :param inverse_temperature: lambda, increase over each iteration by the caller.

      :return: (o_mean, o_stdv, o_activation)
      """

      rr_prime = rr * i_activations

      # rr_prime_sum: sum over all input capsule i
      rr_prime_sum = tf.reduce_sum(rr_prime, axis=-3, keep_dims=True, name='rr_prime_sum')

      # o_mean: (24, 6, 6, 1, 32, 16)
      o_mean = tf.reduce_sum(
        rr_prime * votes, axis=-3, keep_dims=True
      ) / rr_prime_sum

      # o_stdv: (24, 6, 6, 1, 32, 16)
      o_stdv = tf.sqrt(
        tf.reduce_sum(
          rr_prime * tf.square(votes - o_mean), axis=-3, keep_dims=True
        ) / rr_prime_sum
      )

      # o_cost_h: (24, 6, 6, 1, 32, 16)
      o_cost_h = (beta_v + tf.log(o_stdv + epsilon)) * rr_prime_sum

      # o_cost: (24, 6, 6, 1, 32, 1)
      # o_activations_cost = (24, 6, 6, 1, 32, 1)
      # yg: This is done for numeric stability.
      # It is the relative variance between each channel determined which one should activate.
      o_cost = tf.reduce_sum(o_cost_h, axis=-1, keep_dims=True)
      o_cost_mean = tf.reduce_mean(o_cost, axis=-2, keep_dims=True)
      o_cost_stdv = tf.sqrt(
        tf.reduce_sum(
          tf.square(o_cost - o_cost_mean), axis=-2, keep_dims=True
        ) / o_cost.get_shape().as_list()[-2]
      )
      o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + epsilon)

      # (24, 6, 6, 1, 32, 1)
      o_activations = tf.sigmoid(
        inverse_temperature * o_activations_cost
      )

      return o_mean, o_stdv, o_activations

    def e_step(o_mean, o_stdv, o_activations, votes):
      """The E-Step in EM Routing.

      :param o_mean: (24, 6, 6, 1, 32, 16)
      :param o_stdv: (24, 6, 6, 1, 32, 16)
      :param o_activations: (24, 6, 6, 1, 32, 1)
      :param votes: (24, 6, 6, 288, 32, 16)

      :return: rr
      """

      o_p_unit0 = - tf.reduce_sum(
        tf.square(votes - o_mean) / (2 * tf.square(o_stdv)), axis=-1, keep_dims=True
      )

      o_p_unit2 = - tf.reduce_sum(
        tf.log(o_stdv + epsilon), axis=-1, keep_dims=True
      )

      # o_p is the probability density of the h-th component of the vote from i to j
      # (24, 6, 6, 1, 32, 16)
      o_p = o_p_unit0 + o_p_unit2

      # rr: (24, 6, 6, 288, 32, 1)
      zz = tf.log(o_activations + epsilon) + o_p
      rr = tf.nn.softmax(
        zz, dim=len(zz.get_shape().as_list())-2
      )

      return rr

    # inverse_temperature schedule (min, max)
    it_min = 1.0
    it_max = min(iterations, 3.0)
    for it in range(iterations):
      inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, iterations - 1.0)
      o_mean, o_stdv, o_activations = m_step(
        rr, votes, i_activations, beta_v, beta_a, inverse_temperature=inverse_temperature
      )
      if it < iterations - 1:
        rr = e_step(
          o_mean, o_stdv, o_activations, votes
        )

    # pose: (N, OH, OW, o 4 x 4) via squeeze o_mean (24, 6, 6, 32, 16)
    poses = tf.squeeze(o_mean, axis=-3)

    # activation: (N, OH, OW, o) via squeeze o_activationis [24, 6, 6, 32]
    activations = tf.squeeze(o_activations, axis=[-3, -1])

  return poses, activations

# ------------------------------------------------------------------------------#

