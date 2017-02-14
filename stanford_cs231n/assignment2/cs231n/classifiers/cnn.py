import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.bn_params = {}
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # Initialize weights and biases for the three-layer convolutional          #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim

    HH = WW = filter_size
    stride_conv = 1
    pad = (filter_size - 1) / 2
    H_conv = (H + 2 * pad - HH) / stride_conv + 1
    W_conv = (W + 2 * pad - WW) / stride_conv + 1

    width_pool = 2
    height_pool = 2
    stride_pool = 2
    H_pool = (H_conv - height_pool) / stride_pool + 1
    W_pool = (W_conv - width_pool) / stride_pool + 1

    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, HH, WW)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(num_filters * H_pool * W_pool, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    if self.use_batchnorm:
        bn_param1 = {'mode': 'train',
                     'running_mean': np.zeros(num_filters),
                     'running_var': np.zeros(num_filters)}
        gamma1 = np.ones(num_filters)
        beta1 = np.zeros(num_filters)

        bn_param2 = {'mode': 'train',
                     'running_mean': np.zeros(hidden_dim),
                     'running_var': np.zeros(hidden_dim)}
        gamma2 = np.ones(hidden_dim)
        beta2 = np.zeros(hidden_dim)

        self.bn_params['bn_param1'] = bn_param1
        self.bn_params['bn_param2'] = bn_param2

        self.params['beta1'] = beta1
        self.params['beta2'] = beta2
        self.params['gamma1'] = gamma1
        self.params['gamma2'] = gamma2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    if self.use_batchnorm:
        mode = 'test' if y is None else 'train'
        for key, bn_param in self.bn_params.iteritems():
            bn_param[mode] = mode

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    if self.use_batchnorm:
        bn_param1, gamma1, beta1 = self.bn_params['bn_param1'], self.params['gamma1'], self.params['beta1']
        bn_param2, gamma2, beta2 = self.bn_params['bn_param2'], self.params['gamma2'], self.params['beta2']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # Implement the forward pass for the three-layer convolutional net,        #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    from fc_net import affine_batchnorm_relu_forward
    if self.use_batchnorm:
        conv, cache_conv = conv_batchnorm_relu_pool_forward(X, W1, b1, conv_param, pool_param, gamma1, beta1, bn_param1)
        relu, cache_relu = affine_batchnorm_relu_forward(conv, W2, b2, gamma2, beta2, bn_param2)
    else:
        conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        relu, cache_relu = affine_relu_forward(conv, W2, b2)
    scores, cache_scores = affine_forward(relu, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # Implement the backward pass for the three-layer convolutional net,       #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    loss = data_loss + 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))

    grads = {}
    drelu, dW3, db3 = affine_backward(dscores, cache_scores)

    from fc_net import affine_batchnorm_relu_backward

    if self.use_batchnorm:
        dpool, dW2, db2, dgamma2, dbeta2 = affine_batchnorm_relu_backward(
            drelu, cache_relu)
    else:
        dpool, dW2, db2 = affine_relu_backward(drelu, cache_relu)

    if self.use_batchnorm:
        dx, dW1, db1, dgamma1, dbeta1 = conv_batchnorm_relu_pool_backward(
            dpool, cache_conv)
    else:
        dx, dW1, db1 = conv_relu_pool_backward(dpool, cache_conv)

    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3

    grads['W1'], grads['W2'], grads['W3'] = dW1, dW2, dW3
    grads['b1'], grads['b2'], grads['b3'] = db1, db2, db3

    if self.use_batchnorm:
        grads['beta1'], grads['beta2'], grads['gamma1'], grads['gamma2'] = dbeta1, dbeta2, dgamma1, dgamma2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

def conv_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
    """
    A convenience layer that performs a convolution followed by a batch norm and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
    norm, norm_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
    out, relu_cache = relu_forward(norm)

    cache = (conv_cache, norm_cache, relu_cache)

    return out, cache


def conv_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the conv-batchnorm-relu-pool convenience layer
    """
    conv_cache, norm_cache, relu_cache = cache

    drelu = relu_backward(dout, relu_cache)
    dnorm, dgamma, dbeta = spatial_batchnorm_backward(drelu, norm_cache)
    dx, dw, db = conv_backward_fast(dnorm, conv_cache)

    return dx, dw, db, dgamma, dbeta


def conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """

    conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
    norm, norm_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
    relu, relu_cache = relu_forward(norm)
    out, pool_cache = max_pool_forward_fast(relu, pool_param)

    cache = (conv_cache, norm_cache, relu_cache, pool_cache)

    return out, cache


def conv_batchnorm_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, norm_cache, relu_cache, pool_cache = cache

    dpool = max_pool_backward_fast(dout, pool_cache)
    drelu = relu_backward(dpool, relu_cache)
    dnorm, dgamma, dbeta = spatial_batchnorm_backward(drelu, norm_cache)
    dx, dw, db = conv_backward_fast(dnorm, conv_cache)

    return dx, dw, db, dgamma, dbeta

