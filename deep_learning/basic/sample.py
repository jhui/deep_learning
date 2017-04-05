import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_forward(x, w, b):
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout
    dx[x < 0] = 0
    return dx

def softmax_loss(x, y):
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32):
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        layers = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W%d' % (i + 1)] = np.random.randn(layers[i], layers[i + 1]) * weight_scale
            self.params['b%d' % (i + 1)] = np.zeros(layers[i + 1])

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


def loss(self, X, y=None):
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    layer = [None] * (1 + self.num_layers)
    cache_layer = [None] * (1 + self.num_layers)

    layer[0] = X

    for i in range(1, self.num_layers):
        W = self.params['W%d' % i]
        b = self.params['b%d' % i]
        layer[i], cache_layer[i] = affine_relu_forward(layer[i - 1], W, b)

    last_W_name = 'W%d' % self.num_layers
    last_b_name = 'b%d' % self.num_layers
    scores, cache_scores = affine_forward(layer[self.num_layers - 1],
                                          self.params[last_W_name],
                                          self.params[last_b_name])

    if mode == 'test':
        return scores

    loss, grads = 0.0, {}
    loss, dscores = softmax_loss(scores, y)

    for i in range(self.num_layers):
        loss += 0.5 * self.reg * np.sum(self.params['W%d' % (i + 1)] ** 2)

    dx = [None] * (1 + self.num_layers)
    dx[self.num_layers], grads[last_W_name], grads[last_b_name] = affine_backward(dscores, cache_scores)
    grads[last_W_name] += self.reg * self.params[last_W_name]

    for i in reversed(range(1, self.num_layers)):
        dx[i], grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(dx[i + 1], cache_layer[i])
        grads['W%d' % i] += self.reg * self.params['W%d' % i]

    return loss, grads



model = FullyConnectedNet([100, 50, 25],
              weight_scale=5e-2, dtype=np.float64)
