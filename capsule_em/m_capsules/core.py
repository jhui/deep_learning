"""An implementation of matrix capsules with EM routing.
"""

from tensorflow.contrib.layers.python.layers import initializers

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

epsilon = 1e-9

def conv2d(inputs, kernel, out_channels, stride, padding, name, is_train=True, activation_fn=None):
  with slim.arg_scope([slim.conv2d], trainable=is_train):
    with tf.variable_scope(name) as scope:
      output = slim.conv2d(inputs,
                           num_outputs=out_channels,
                           kernel_size=[kernel, kernel], stride=stride, padding=padding,
                           scope=scope, activation_fn=activation_fn)
      tf.logging.info(f"{name} output shape: {output.get_shape()}")

  return output


def primary_caps(inputs, kernel_size, out_capsules, stride, padding, pose_shape, name):
    """This constructs a primary capsule layer using regular convolution layer.

    :param inputs: shape (N, H, W, C) (?, 14, 14, 32)
    :param kernel_size: Apply a filter of [kernel, kernel] [5x5]
    :param out_capsules: # of output capsule (32)
    :param stride: 1, 2, or ... (1)
    :param padding: padding: SAME or VALID.
    :param pose_shape: (4, 4)
    :param name: scope name

    :return: (poses, activations), (poses (?, 14, 14, 32, 4, 4), activations (?, 14, 14, 32))
    """

    with tf.variable_scope(name) as scope:
        # Generate the poses matrics for the 32 output capsules
        poses = conv2d(
            inputs,
            kernel_size, out_capsules * pose_shape[0] * pose_shape[1], stride, padding=padding,
            name='pose_stacked'
        )

        input_shape = inputs.get_shape()

        # Reshape 16 scalar values into a 4x4 matrix
        poses = tf.reshape(
            poses, shape=[-1, input_shape[-3], input_shape[-2], out_capsules, pose_shape[0], pose_shape[1]],
            name='poses'
        )

        # Generate the activation for the 32 output capsules
        activations = conv2d(
            inputs,
            kernel_size,
            out_capsules,
            stride,
            padding=padding,
            activation_fn=tf.sigmoid,
            name='activation'
        )

        tf.summary.histogram(
            'activations', activations
        )

    # poses (?, 14, 14, 32, 4, 4), activations (?, 14, 14, 32)
    return poses, activations

def kernel_tile(input, kernel, stride):
    """This constructs a primary capsule layer using regular convolution layer.

    :param inputs: shape (?, 14, 14, 32, 4, 4)
    :param kernel: 3
    :param stride: 2

    :return output: (50, 5, 5, 3x3=9, 136)
    """

    # (?, 14, 14, 32x(16)=512)
    input_shape = input.get_shape()
    size = input_shape[4]*input_shape[5] if len(input_shape)>5 else 1
    input = tf.reshape(input, shape=[-1, input_shape[1], input_shape[2], input_shape[3]*size])


    input_shape = input.get_shape()
    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                  kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0 # (3, 3, 512, 9)

    # (3, 3, 512, 9)
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)

    # (?, 6, 6, 4608)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                    1, stride, stride, 1], padding='VALID')

    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[-1, output_shape[1], output_shape[2], input_shape[3], kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    # (?, 6, 6, 9, 512)
    return output

def mat_transform(input, output_cap_size, size):
    """Compute the vote.

    :param inputs: shape (size, 288, 16)
    :param output_cap_size: 32

    :return votes: (24, 5, 5, 3x3=9, 136)
    """

    caps_num_i = int(input.get_shape()[1]) # 288
    output = tf.reshape(input, shape=[size, caps_num_i, 1, 4, 4]) # (size, 288, 1, 4, 4)

    w = slim.variable('w', shape=[1, caps_num_i, output_cap_size, 4, 4], dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0)) # (1, 288, 32, 4, 4)
    w = tf.tile(w, [size, 1, 1, 1, 1])  # (24, 288, 32, 4, 4)

    output = tf.tile(output, [1, 1, output_cap_size, 1, 1]) # (size, 288, 32, 4, 4)

    votes = tf.matmul(output, w) # (24, 288, 32, 4, 4)
    votes = tf.reshape(votes, [size, caps_num_i, output_cap_size, 16]) # (size, 288, 32, 16)

    return votes

def coord_addition(votes, H, W):
    """Coordinate addition.

    :param votes: (24, 4, 4, 32, 10, 16)
    :param H, W: spaital height and width 4

    :return votes: (24, 4, 4, 32, 10, 16)
    """
    coordinate_offset_hh = tf.reshape(
      (tf.range(H, dtype=tf.float32) + 0.50) / H, [1, H, 1, 1, 1]
    )
    coordinate_offset_h0 = tf.constant(
      0.0, shape=[1, H, 1, 1, 1], dtype=tf.float32
    )
    coordinate_offset_h = tf.stack(
      [coordinate_offset_hh, coordinate_offset_h0] + [coordinate_offset_h0 for _ in range(14)], axis=-1
    )  # (1, 4, 1, 1, 1, 16)

    coordinate_offset_ww = tf.reshape(
      (tf.range(W, dtype=tf.float32) + 0.50) / W, [1, 1, W, 1, 1]
    )
    coordinate_offset_w0 = tf.constant(
      0.0, shape=[1, 1, W, 1, 1], dtype=tf.float32
    )
    coordinate_offset_w = tf.stack(
      [coordinate_offset_w0, coordinate_offset_ww] + [coordinate_offset_w0 for _ in range(14)], axis=-1
    ) # (1, 1, 4, 1, 1, 16)

    # (24, 4, 4, 32, 10, 16)
    votes = votes + coordinate_offset_h + coordinate_offset_w

    return votes

def conv_capsule(inputs, shape, strides, iterations, batch_size, name):
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
  :param strides: often [1, 2, 2, 1] (stride 2), or [1, 1, 1, 1] (stride 1).
  :param iterations: number of iterations in EM routing. 3
  :param name: name.

  :return: (poses, activations).

  """
  inputs_poses, inputs_activations = inputs

  with tf.variable_scope(name) as scope:

      stride = strides[1] # 2
      i_size = shape[-2] # 32
      o_size = shape[-1] # 32
      pose_size = inputs_poses.get_shape()[-1]  # 4

      # Tile the input capusles' pose matrices to the spatial dimension of the output capsules
      # Such that we can later multiple with the transformation matrices to generate the votes.
      inputs_poses = kernel_tile(inputs_poses, 3, stride)  # (?, 14, 14, 32, 4, 4) -> (?, 6, 6, 3x3=9, 32x16=512)

      # Tile the activations needed for the EM routing
      inputs_activations = kernel_tile(inputs_activations, 3, stride)  # (?, 14, 14, 32) -> (?, 6, 6, 9, 32)
      spatial_size = int(inputs_activations.get_shape()[1]) # 6

      # Reshape it for later operations
      inputs_poses = tf.reshape(inputs_poses, shape=[-1, 3 * 3 * i_size, 16])  # (?, 9x32=288, 16)
      inputs_activations = tf.reshape(inputs_activations, shape=[-1, spatial_size, spatial_size, 3 * 3 * i_size]) # (?, 6, 6, 9x32=288)

      with tf.variable_scope('votes') as scope:

          # Generate the votes by multiply it with the transformation matrices
          votes = mat_transform(inputs_poses, o_size, size=batch_size*spatial_size*spatial_size)  # (864, 288, 32, 16)

          # Reshape the vote for EM routing
          votes_shape = votes.get_shape()
          votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, votes_shape[-3], votes_shape[-2], votes_shape[-1]]) # (24, 6, 6, 288, 32, 16)
          tf.logging.info(f"{name} votes shape: {votes.get_shape()}")

      with tf.variable_scope('routing') as scope:

          # beta_v and beta_a one for each output capsule: (1, 1, 1, 32)
          beta_v = tf.get_variable(
              name='beta_v', shape=[1, 1, 1, o_size], dtype=tf.float32,
              initializer=initializers.xavier_initializer()
          )
          beta_a = tf.get_variable(
              name='beta_a', shape=[1, 1, 1, o_size], dtype=tf.float32,
              initializer=initializers.xavier_initializer()
          )

          # Use EM routing to compute the pose and activation
          # votes (24, 6, 6, 3x3x32=288, 32, 16), inputs_activations (?, 6, 6, 288)
          # poses (24, 6, 6, 32, 16), activation (24, 6, 6, 32)
          poses, activations = matrix_capsules_em_routing(
              votes, inputs_activations, beta_v, beta_a, iterations, name='em_routing'
          )

          # Reshape it back to 4x4 pose matrix
          poses_shape = poses.get_shape()
          # (24, 6, 6, 32, 4, 4)
          poses = tf.reshape(
              poses, [
                  poses_shape[0], poses_shape[1], poses_shape[2], poses_shape[3], pose_size, pose_size
              ]
          )

      tf.logging.info(f"{name} pose shape: {poses.get_shape()}")
      tf.logging.info(f"{name} activations shape: {activations.get_shape()}")

      return poses, activations

def class_capsules(inputs, num_classes, iterations, batch_size, name):
    """
    :param inputs: ((24, 4, 4, 32, 4, 4), (24, 4, 4, 32))
    :param num_classes: 10
    :param iterations: 3
    :param batch_size: 24
    :param name:
    :return poses, activations: poses (24, 10, 4, 4), activation (24, 10).
    """

    inputs_poses, inputs_activations = inputs # (24, 4, 4, 32, 4, 4), (24, 4, 4, 32)

    inputs_shape = inputs_poses.get_shape()
    spatial_size = int(inputs_shape[1])  # 4
    pose_size = int(inputs_shape[-1])    # 4
    i_size = int(inputs_shape[3])        # 32

    # inputs_poses (24*4*4=384, 32, 16)
    inputs_poses = tf.reshape(inputs_poses, shape=[batch_size*spatial_size*spatial_size, inputs_shape[-3], inputs_shape[-2]*inputs_shape[-2] ])

    with tf.variable_scope(name) as scope:
        with tf.variable_scope('votes') as scope:
            # inputs_poses (384, 32, 16)
            # votes: (384, 32, 10, 16)
            votes = mat_transform(inputs_poses, num_classes, size=batch_size*spatial_size*spatial_size)
            tf.logging.info(f"{name} votes shape: {votes.get_shape()}")

            # votes (24, 4, 4, 32, 10, 16)
            votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, i_size, num_classes, pose_size*pose_size])

            # (24, 4, 4, 32, 10, 16)
            votes = coord_addition(votes, spatial_size, spatial_size)

            tf.logging.info(f"{name} votes shape with coord addition: {votes.get_shape()}")

        with tf.variable_scope('routing') as scope:
            # beta_v and beta_a one for each output capsule: (1, 10)
            beta_v = tf.get_variable(
                name='beta_v', shape=[1, num_classes], dtype=tf.float32,
                initializer=initializers.xavier_initializer()
            )
            beta_a = tf.get_variable(
                name='beta_a', shape=[1, num_classes], dtype=tf.float32,
                initializer=initializers.xavier_initializer()
            )

            # votes (24, 4, 4, 32, 10, 16) -> (24, 512, 10, 16)
            votes_shape = votes.get_shape()
            votes = tf.reshape(votes, shape=[batch_size, votes_shape[1] * votes_shape[2] * votes_shape[3], votes_shape[4], votes_shape[5]] )

            # inputs_activations (24, 4, 4, 32) -> (24, 512)
            inputs_activations = tf.reshape(inputs_activations, shape=[batch_size,
                                                                       votes_shape[1] * votes_shape[2] * votes_shape[3]])

            # votes (24, 512, 10, 16), inputs_activations (24, 512)
            # poses (24, 10, 16), activation (24, 10)
            poses, activations = matrix_capsules_em_routing(
                votes, inputs_activations, beta_v, beta_a, iterations, name='em_routing'
            )

        # poses (24, 10, 16) -> (24, 10, 4, 4)
        poses = tf.reshape(poses, shape=[batch_size, num_classes, pose_size, pose_size] )

        # poses (24, 10, 4, 4), activation (24, 10)
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

    # Match rr (routing assignment) shape, i_activations shape with votes shape for broadcasting in EM routing

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

      # Mean of the output capsules: o_mean(24, 6, 6, 1, 32, 16)
      o_mean = tf.reduce_sum(
        rr_prime * votes, axis=-3, keep_dims=True
      ) / rr_prime_sum

      # Standard deviation of the output capsule:  o_stdv (24, 6, 6, 1, 32, 16)
      o_stdv = tf.sqrt(
        tf.reduce_sum(
          rr_prime * tf.square(votes - o_mean), axis=-3, keep_dims=True
        ) / rr_prime_sum
      )

      # o_cost_h: (24, 6, 6, 1, 32, 16)
      o_cost_h = (beta_v + tf.log(o_stdv + epsilon)) * rr_prime_sum

      # o_cost: (24, 6, 6, 1, 32, 1)
      # o_activations_cost = (24, 6, 6, 1, 32, 1)
      # For numeric stability.
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


