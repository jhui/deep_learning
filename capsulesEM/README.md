## Credit

This work is dervied from [github repository](https://github.com/gyang274/capsulesEM)

# Capsule

A Tensorflow Implementation of Hinton's __[Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)__.

## Quick Start

```
$ git clone https://github.com/gyang274/capsulesEM.git && cd capsulesEM

$ cd src

$ python train.py

# open a new terminal (ctrl + alt + t)

$ python tests.py
```

Note: 

1. Tensorflow v1.4.0.

2. This `train.py` and `tests.py` assumes the user have 2 GPU card: `train.py` will use the first GPU card, and `tests.py` will use the second one. In case a different setting required, or multiple GPUs are available for training, modify `visible_device_list` in `session_config` in `slim.learning.train()` in `train.py`, or modify `visible_device_list` in `session_config` in `slim.evaluation.evaluation_loop()` in `tests.py`.

## Status

### MNIST

1. (R0I1) Network architecture same as in paper, Matrix Capsules with EM Routing, Figure 1.

    - Spread loss only, no reconstruction loss.
    
    - Adam Optimizer, learning rate default 0.001, no learning rate decay. 
    
    - Batch size 24 (due to limit of GPU memory), iteration 1. 
    
    - GPU: half K80 12GB memory, 2s-3s per training step.

    - Step: 43942, Test Accuracy: __99.37%__.

    ![Screenshot Tensorboard](doc/src/fig/capsulesEM-V0-R0I1-screenshot-eval-loss.png)
    
    __Remark__: Because of `allow_smaller_final_batch=False` and `batch_size=24`, test is running on a random sample 9984 of 10000, so worse case test accuracy could be 99.21%. Modify the `src/datasets/mnist.py` and `src/test.py` to run test on full test dataset.

1. (R0I2) As above, except iteration 2. (TODO)

1. (R1I2) As above, add reconstruction loss, iteration 2. (TODO)

## Matrix Capsules Nets and Layers

Build a matrix capsules neural network as the same way of building CNN:

```
def capsules_net(inputs, num_classes, iterations, name='CapsuleEM-V0'):
  """Replicate the network in `Matrix Capsules with EM Routing.`
  """

  with tf.variable_scope(name) as scope:

    # inputs [N, H, W, C] -> conv2d, 5x5, strides 2, channels 32 -> nets [N, OH, OW, 32]
    nets = _conv2d_wrapper(
      inputs, shape=[5, 5, 1, 32], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu, name='conv1'
    )
    # inputs [N, H, W, C] -> conv2d, 1x1, strides 1, channels 32x(4x4+1) -> (poses, activations)
    nets = capsules_init(
      nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], padding='VALID', pose_shape=[4, 4], name='capsule_init'
    )
    # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 2 -> (poses, activations)
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv1'
    )
    # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 1 -> (poses, activations)
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], iterations=iterations, name='capsule_conv2'
    )
    # inputs: (poses, activations) -> capsule-fc 1x1x32x10x4x4 shared view transform matrix within each channel -> (poses, activations)
    nets = capsules_fc(
      nets, num_classes, iterations=iterations, name='capsule_fc'
    )

    poses, activations = nets

  return poses, activations
```

In particular,

- `capsules_init()` takes a CNN layer as inputs, and produces a matrix capsule layer (e.g., primaryCaps) as output. 

    This operation is corresponding to the layer `A -> B` in the paper.

- `capsules_conv()` takes a matrix capsule layer (e.g., primaryCaps, ConvCaps1) as inputs, and produces a matrix capsule layer (e.g., ConvCaps1, ConvCaps2) as output.

    This operation is corresponding to the layer `B -> C` and `C -> D` in the paper.
 
- `capsules_fc()` takes a matrix capsule layer (e.g., ConvCaps2) as inputs, and produces an output matrix capsule layer with poses and activations (e.g., Class Capsules) as output. 
    
    This operation is correponding to the layer `D -> E` in the paper. 

## TODO

1. How `tf.stop_gradient()` in EM? How iteration > 1 cause NaN in loss and capsules_init() activations?

1. Add `learning_rate decay` in `train.py`

1. Add train.py/tests.py on smallNORB.

## Questions

1. $$\lambda$$ schedule is never mentioned in paper.

1. The place encode in lower level and rate encode in higher level is not discussed, other than a coordinate addition in last layer.

## GitHub Page

This [`gh-pages`](https://gyang274.github.io/capsulesEM/) includes all notes.

## GitHub Repository

This [github repository](https://github.com/gyang274/capsulesEM) includes all source codes.
