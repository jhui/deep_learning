# As usual, a bit of setup

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

do_plotting = False
do_training = False

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
out, _ = conv_forward_naive(x, w, b, conv_param)
correct_out = np.array([[[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]]])

# Compare your output to ours; difference should be around 1e-8
print 'Testing conv_forward_naive'
print 'difference: ', rel_error(out, correct_out)

from scipy.misc import imread, imresize

kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
# kitten is wide, and puppy is already square
d = kitten.shape[1] - kitten.shape[0]
kitten_cropped = kitten[:, d/2:-d/2, :]

img_size = 200   # Make this smaller if it runs too slow
x = np.zeros((2, 3, img_size, img_size))
x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

# Set up a convolutional weights holding 2 filters, each 3x3
w = np.zeros((2, 3, 3, 3))

# The first filter converts the image to grayscale.
# Set up the red, green, and blue channels of the filter.
w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

# Second filter detects horizontal edges in the blue channel.
w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Vector of biases. We don't need any bias for the grayscale
# filter, but for the edge detection filter we want to add 128
# to each output so that nothing is negative.
b = np.array([0, 128])

# Compute the result of convolving each input in x with each filter in w,
# offsetting by b, and storing the results in out.
out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})

def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')

if do_plotting:
    # Show the original images and the results of the conv operation
    plt.subplot(2, 3, 1)
    imshow_noax(puppy, normalize=False)
    plt.title('Original image')
    plt.subplot(2, 3, 2)
    imshow_noax(out[0, 0])
    plt.title('Grayscale')
    plt.subplot(2, 3, 3)
    imshow_noax(out[0, 1])
    plt.title('Edges')
    plt.subplot(2, 3, 4)
    imshow_noax(kitten_cropped, normalize=False)
    plt.subplot(2, 3, 5)
    imshow_noax(out[1, 0])
    plt.subplot(2, 3, 6)
    imshow_noax(out[1, 1])
    plt.show()

x = np.random.randn(4, 3, 5, 5)
w = np.random.randn(2, 3, 3, 3)
b = np.random.randn(2,)
dout = np.random.randn(4, 2, 5, 5)
conv_param = {'stride': 1, 'pad': 1}

dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

out, cache = conv_forward_naive(x, w, b, conv_param)
dx, dw, db = conv_backward_naive(dout, cache)

# Your errors should be around 1e-9'
print 'Testing conv_backward_naive function'
print 'dx error: ', rel_error(dx, dx_num)
print 'dw error: ', rel_error(dw, dw_num)
print 'db error: ', rel_error(db, db_num)

x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = max_pool_forward_naive(x, pool_param)

correct_out = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])

# Compare your output with ours. Difference should be around 1e-8.
print 'Testing max_pool_forward_naive function:'
print 'difference: ', rel_error(out, correct_out)

x = np.random.randn(3, 2, 8, 8)
dout = np.random.randn(3, 2, 4, 4)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)

out, cache = max_pool_forward_naive(x, pool_param)
dx = max_pool_backward_naive(dout, cache)

# Your error should be around 1e-12
print 'Testing max_pool_backward_naive function:'
print 'dx error: ', rel_error(dx, dx_num)

from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time

x = np.random.randn(100, 3, 31, 31)
w = np.random.randn(25, 3, 3, 3)
b = np.random.randn(25,)
dout = np.random.randn(100, 25, 16, 16)
conv_param = {'stride': 2, 'pad': 1}

t0 = time()
out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
t1 = time()
out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
t2 = time()

print 'Testing conv_forward_fast:'
print 'Naive: %fs' % (t1 - t0)
print 'Fast: %fs' % (t2 - t1)
print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
print 'Difference: ', rel_error(out_naive, out_fast)

t0 = time()
dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
t1 = time()
dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
t2 = time()

print '\nTesting conv_backward_fast:'
print 'Naive: %fs' % (t1 - t0)
print 'Fast: %fs' % (t2 - t1)
print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
print 'dx difference: ', rel_error(dx_naive, dx_fast)
print 'dw difference: ', rel_error(dw_naive, dw_fast)
print 'db difference: ', rel_error(db_naive, db_fast)

from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast

x = np.random.randn(100, 3, 32, 32)
dout = np.random.randn(100, 3, 16, 16)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

t0 = time()
out_naive, cache_naive = max_pool_forward_naive(x, pool_param)
t1 = time()
out_fast, cache_fast = max_pool_forward_fast(x, pool_param)
t2 = time()

print 'Testing pool_forward_fast:'
print 'Naive: %fs' % (t1 - t0)
print 'fast: %fs' % (t2 - t1)
print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
print 'difference: ', rel_error(out_naive, out_fast)

t0 = time()
dx_naive = max_pool_backward_naive(dout, cache_naive)
t1 = time()
dx_fast = max_pool_backward_fast(dout, cache_fast)
t2 = time()

print '\nTesting pool_backward_fast:'
print 'Naive: %fs' % (t1 - t0)
print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
print 'dx difference: ', rel_error(dx_naive, dx_fast)

from cs231n.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward

x = np.random.randn(2, 3, 16, 16)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)
dx, dw, db = conv_relu_pool_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b, dout)

print 'Testing conv_relu_pool'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)

from cs231n.layer_utils import conv_relu_forward, conv_relu_backward

x = np.random.randn(2, 3, 8, 8)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}

out, cache = conv_relu_forward(x, w, b, conv_param)
dx, dw, db = conv_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, w, b, conv_param)[0], b, dout)

print 'Testing conv_relu:'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)

model = ThreeLayerConvNet()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print 'Initial loss (no regularization): ', loss

model.reg = 0.5
loss, grads = model.loss(X, y)
print 'Initial loss (with regularization): ', loss

if do_training:
    num_inputs = 2
    input_dim = (3, 16, 16)
    reg = 0.0
    num_classes = 10
    X = np.random.randn(num_inputs, *input_dim)
    y = np.random.randint(num_classes, size=num_inputs)

    model = ThreeLayerConvNet(num_filters=3, filter_size=3,
                              input_dim=input_dim, hidden_dim=7,
                              dtype=np.float64)
    loss, grads = model.loss(X, y)
    for param_name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
        e = rel_error(param_grad_num, grads[param_name])
        print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

    num_train = 100
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    # (Iteration 20 / 20) loss: 0.649341
    # (Epoch 10 / 10) train acc: 0.920000; val_acc: 0.237000
    # Train small data set
    model = ThreeLayerConvNet(weight_scale=1e-2)
    solver = Solver(model, small_data,
                    num_epochs=10, batch_size=50,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=1)
    solver.train()
    print ('Train small data set')
    if do_plotting:
        plt.subplot(2, 1, 1)
        plt.plot(solver.loss_history, 'o')
        plt.xlabel('iteration')
        plt.ylabel('loss')

        plt.subplot(2, 1, 2)
        plt.plot(solver.train_acc_history, '-o')
        plt.plot(solver.val_acc_history, '-o')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()

    # (Iteration 961 / 980) loss: 1.390810
    # (Epoch 1 / 1) train acc: 0.489000; val_acc: 0.505000
    # Train regular data for 1 Epoch
    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)
    solver = Solver(model, data,
                    num_epochs=1, batch_size=50,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()
    print 'Train regular data for 1 Epoch'

    from cs231n.vis_utils import visualize_grid

    if do_plotting:
        grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
        plt.imshow(grid.astype('uint8'))
        plt.axis('off')
        plt.gcf().set_size_inches(5, 5)
        plt.show()


# Check the training-time forward pass by checking means and variances
# of features both before and after spatial batch normalization

N, C, H, W = 2, 3, 4, 5
x = 4 * np.random.randn(N, C, H, W) + 10

print 'Before spatial batch normalization:'
print '  Shape: ', x.shape
print '  Means: ', x.mean(axis=(0, 2, 3))
print '  Stds: ', x.std(axis=(0, 2, 3))

# Means should be close to zero and stds close to one
gamma, beta = np.ones(C), np.zeros(C)
bn_param = {'mode': 'train'}
out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
print 'After spatial batch normalization:'
print '  Shape: ', out.shape
print '  Means: ', out.mean(axis=(0, 2, 3))
print '  Stds: ', out.std(axis=(0, 2, 3))

# Means should be close to beta and stds close to gamma
gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
print 'After spatial batch normalization (nontrivial gamma, beta):'
print '  Shape: ', out.shape
print '  Means: ', out.mean(axis=(0, 2, 3))
print '  Stds: ', out.std(axis=(0, 2, 3))

# Check the test-time forward pass by running the training-time
# forward pass many times to warm up the running averages, and then
# checking the means and variances of activations after a test-time
# forward pass.

N, C, H, W = 10, 4, 11, 12

bn_param = {'mode': 'train'}
gamma = np.ones(C)
beta = np.zeros(C)
for t in xrange(50):
    x = 2.3 * np.random.randn(N, C, H, W) + 13
    spatial_batchnorm_forward(x, gamma, beta, bn_param)
bn_param['mode'] = 'test'
x = 2.3 * np.random.randn(N, C, H, W) + 13
a_norm, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)

# Means should be close to zero and stds close to one, but will be
# noisier than training-time forward passes.
print 'After spatial batch normalization (test-time):'
print '  means: ', a_norm.mean(axis=(0, 2, 3))
print '  stds: ', a_norm.std(axis=(0, 2, 3))

N, C, H, W = 2, 3, 4, 5
x = 5 * np.random.randn(N, C, H, W) + 12
gamma = np.random.randn(C)
beta = np.random.randn(C)
dout = np.random.randn(N, C, H, W)

bn_param = {'mode': 'train'}
fx = lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
fg = lambda a: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
fb = lambda b: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
da_num = eval_numerical_gradient_array(fg, gamma, dout)
db_num = eval_numerical_gradient_array(fb, beta, dout)

_, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
print 'dx error: ', rel_error(dx_num, dx)
print 'dgamma error: ', rel_error(da_num, dgamma)
print 'dbeta error: ', rel_error(db_num, dbeta)

if do_training:
    # (Iteration 961 / 980) loss: 1.699363
    # (Epoch 1 / 1) train acc: 0.464000; val_acc: 0.454000
    # Base line
    model_base = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

    solver_base = Solver(model_base, data,
                    num_epochs=1, batch_size=50,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver_base.train()
    print 'Base line'

    # (Iteration 961 / 980) loss: 1.292299
    # (Epoch 1 / 1) train acc: 0.523000; val_acc: 0.478000
    # filter size=5
    model_base = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size=5)

    solver_base = Solver(model_base, data,
                    num_epochs=1, batch_size=50,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver_base.train()
    print 'filter size=5'

    # (Iteration 961 / 980) loss: 1.444363
    # (Epoch 1 / 1) train acc: 0.555000; val_acc: 0.543000
    # filter size=3
    model_base = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size=3)

    solver_base = Solver(model_base, data,
                    num_epochs=1, batch_size=50,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver_base.train()
    print 'filter size=3'

    # (Iteration 961 / 980) loss: 1.301396
    # (Epoch 1 / 1) train acc: 0.576000; val_acc: 0.532000
    # filters=16
    model_base = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size=3, num_filters=16)

    solver_base = Solver(model_base, data,
                     num_epochs=1, batch_size=50,
                     update_rule='adam',
                     optim_config={
                       'learning_rate': 1e-3,
                     },
                     verbose=True, print_every=20)
    solver_base.train()
    print 'filters=16'

    # (Iteration 961 / 980) loss: 1.616242
    # (Epoch 1 / 1) train acc: 0.489000; val_acc: 0.495000
    # filters=64
    model_base = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size=3, num_filters=64)

    solver_base = Solver(model_base, data,
                     num_epochs=1, batch_size=50,
                     update_rule='adam',
                     optim_config={
                       'learning_rate': 1e-3,
                     },
                     verbose=True, print_every=20)
    solver_base.train()
    print 'filters=64'

from cs231n.classifiers.cnn import conv_batchnorm_relu_forward, conv_batchnorm_relu_backward

x = np.random.randn(2, 3, 16, 16)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
conv_param = {'stride': 1, 'pad': 1}

C = x.shape[1]
gamma = np.random.randn(C)
beta = np.random.randn(C)
bn_param = {'mode': 'train',
            'running_mean': np.zeros(C),
            'running_var': np.zeros(C)}

dout = np.random.randn(2, 3, 16, 16)

out, cache = conv_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)
dx, dw, db, dgamma, dbeta = conv_batchnorm_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], b, dout)
dgamma_num = eval_numerical_gradient_array(lambda gamma:conv_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], gamma, dout)
dbeta_num = eval_numerical_gradient_array(lambda beta:conv_batchnorm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param)[0], beta, dout)

print 'Testing conv_batchnorm_relu_backward'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)
print 'dbeta error: ', rel_error(dbeta_num, dbeta)
print 'dgamma error: ', rel_error(dgamma_num, dgamma)

from cs231n.classifiers.cnn import conv_batchnorm_relu_pool_forward, conv_batchnorm_relu_pool_backward

x = np.random.randn(2, 3, 16, 16)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
C = x.shape[1]
gamma = np.random.randn(C)
beta = np.random.randn(C)
bn_param = {'mode': 'train',
            'running_mean': np.zeros(C),
            'running_var': np.zeros(C)}

dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

out, cache = conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)
dx, dw, db, dgamma, dbeta = conv_batchnorm_relu_pool_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], b, dout)
dgamma_num = eval_numerical_gradient_array(lambda gamma:conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], gamma, dout)
dbeta_num = eval_numerical_gradient_array(lambda beta:conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param)[0], beta, dout)

print 'Testing conv_batchnorm_relu_pool_backward'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)
print 'dbeta error: ', rel_error(dbeta_num, dbeta)
print 'dgamma error: ', rel_error(dgamma_num, dgamma)

num_inputs = 2
input_dim = (3, 16, 16)
reg = 0.0
num_classes = 10
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ThreeLayerConvNet(num_filters=3, filter_size=3,
                          input_dim=input_dim, hidden_dim=7,
                          dtype=np.float64, use_batchnorm = True)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))


if do_training:
    # (Iteration 961 / 980) loss: 1.616239
    # (Epoch 1 / 1) train acc: 0.573000; val_acc: 0.544000
    # Use batchnorm
    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size=3, num_filters=32,
                                   use_batchnorm = True)

    solver_base = Solver(model, data,
                         num_epochs=1, batch_size=50,
                         update_rule='adam',
                         optim_config={
                             'learning_rate': 1e-3,
                         },
                         verbose=True, print_every=20)
    solver_base.train()
    print 'Use batchnorm'

from cs231n.classifiers.convnet import ThreeLayerConvNet2

# (Iteration 11751 / 11760) loss: 0.506156
# (Epoch 12 / 12) train acc: 0.933000; val_acc: 0.774000
# Final training
if True or do_training:
    model = ThreeLayerConvNet2(weight_scale=0.001, hidden_dims=[1024, 256], reg=0.001, filter_size=3, num_filters=[64, 128],
                                   use_batchnorm = True)

    solver_base = Solver(model, data,
                         num_epochs=12, batch_size=50,
                         lr_decay=0.95,
                         update_rule='adam',
                         optim_config={
                             'learning_rate': 1e-3,
                         },
                         verbose=True, print_every=50)
    solver_base.train()
    print 'Final training'

# (Iteration 1 / 11760) loss: 2.305010
# (Epoch 0 / 12) train acc: 0.143000; val_acc: 0.129000
# (Iteration 51 / 11760) loss: 1.956194
# (Iteration 101 / 11760) loss: 1.818976
# (Epoch 0 / 12) train acc: 0.439000; val_acc: 0.451000
# (Iteration 151 / 11760) loss: 1.923948
# (Iteration 201 / 11760) loss: 1.653811
# (Epoch 0 / 12) train acc: 0.463000; val_acc: 0.496000
# (Iteration 251 / 11760) loss: 1.758656
# (Iteration 301 / 11760) loss: 1.721636
# (Epoch 0 / 12) train acc: 0.502000; val_acc: 0.512000
# (Iteration 351 / 11760) loss: 1.648367
# (Iteration 401 / 11760) loss: 1.715918
# (Epoch 0 / 12) train acc: 0.548000; val_acc: 0.567000
# (Iteration 451 / 11760) loss: 1.364693
# (Iteration 501 / 11760) loss: 1.551408
# (Epoch 0 / 12) train acc: 0.604000; val_acc: 0.595000
# (Iteration 551 / 11760) loss: 1.245668
# (Iteration 601 / 11760) loss: 1.326670
# (Epoch 0 / 12) train acc: 0.619000; val_acc: 0.560000
# (Iteration 651 / 11760) loss: 1.500873
# (Iteration 701 / 11760) loss: 1.302782
# (Epoch 0 / 12) train acc: 0.620000; val_acc: 0.601000
# (Iteration 751 / 11760) loss: 1.492216
# (Iteration 801 / 11760) loss: 1.552367
# (Epoch 0 / 12) train acc: 0.623000; val_acc: 0.612000
# (Iteration 851 / 11760) loss: 1.471395
# (Iteration 901 / 11760) loss: 1.622362
# (Epoch 0 / 12) train acc: 0.607000; val_acc: 0.617000
# (Iteration 951 / 11760) loss: 1.460451
# (Epoch 1 / 12) train acc: 0.639000; val_acc: 0.606000
# (Iteration 1001 / 11760) loss: 1.421117
# (Epoch 1 / 12) train acc: 0.627000; val_acc: 0.616000
# (Iteration 1051 / 11760) loss: 1.421810
# (Iteration 1101 / 11760) loss: 1.547926
# (Epoch 1 / 12) train acc: 0.637000; val_acc: 0.646000
# (Iteration 1151 / 11760) loss: 1.730231
# (Iteration 1201 / 11760) loss: 1.343394
# (Epoch 1 / 12) train acc: 0.665000; val_acc: 0.651000
# (Iteration 1251 / 11760) loss: 1.088051
# (Iteration 1301 / 11760) loss: 1.554329
# (Epoch 1 / 12) train acc: 0.658000; val_acc: 0.631000
# (Iteration 1351 / 11760) loss: 1.111654
# (Iteration 1401 / 11760) loss: 1.335437
# (Epoch 1 / 12) train acc: 0.667000; val_acc: 0.609000
# (Iteration 1451 / 11760) loss: 1.862468
# (Iteration 1501 / 11760) loss: 1.401548
# (Epoch 1 / 12) train acc: 0.660000; val_acc: 0.637000
# (Iteration 1551 / 11760) loss: 1.448118
# (Iteration 1601 / 11760) loss: 1.406901
# (Epoch 1 / 12) train acc: 0.646000; val_acc: 0.630000
# (Iteration 1651 / 11760) loss: 1.163941
# (Iteration 1701 / 11760) loss: 1.220936
# (Epoch 1 / 12) train acc: 0.717000; val_acc: 0.663000
# (Iteration 1751 / 11760) loss: 1.133998
# (Iteration 1801 / 11760) loss: 1.445657
# (Epoch 1 / 12) train acc: 0.695000; val_acc: 0.669000
# (Iteration 1851 / 11760) loss: 1.547976
# (Iteration 1901 / 11760) loss: 1.622304
# (Epoch 1 / 12) train acc: 0.726000; val_acc: 0.673000
# (Iteration 1951 / 11760) loss: 1.513368
# (Epoch 2 / 12) train acc: 0.663000; val_acc: 0.677000
# (Iteration 2001 / 11760) loss: 1.491336
# (Epoch 2 / 12) train acc: 0.685000; val_acc: 0.668000
# (Iteration 2051 / 11760) loss: 1.236890
# (Iteration 2101 / 11760) loss: 1.182584
# (Epoch 2 / 12) train acc: 0.696000; val_acc: 0.667000
# (Iteration 2151 / 11760) loss: 1.145401
# (Iteration 2201 / 11760) loss: 1.159258
# (Epoch 2 / 12) train acc: 0.707000; val_acc: 0.678000
# (Iteration 2251 / 11760) loss: 1.134580
# (Iteration 2301 / 11760) loss: 1.369594
# (Epoch 2 / 12) train acc: 0.717000; val_acc: 0.683000
# (Iteration 2351 / 11760) loss: 1.090823
# (Iteration 2401 / 11760) loss: 1.214100
# (Epoch 2 / 12) train acc: 0.730000; val_acc: 0.680000
# (Iteration 2451 / 11760) loss: 1.506453
# (Iteration 2501 / 11760) loss: 1.322179
# (Epoch 2 / 12) train acc: 0.736000; val_acc: 0.687000
# (Iteration 2551 / 11760) loss: 1.337743
# (Iteration 2601 / 11760) loss: 1.133788
# (Epoch 2 / 12) train acc: 0.740000; val_acc: 0.695000
# (Iteration 2651 / 11760) loss: 1.237084
# (Iteration 2701 / 11760) loss: 1.211530
# (Epoch 2 / 12) train acc: 0.733000; val_acc: 0.703000
# (Iteration 2751 / 11760) loss: 1.107685
# (Iteration 2801 / 11760) loss: 1.174298
# (Epoch 2 / 12) train acc: 0.734000; val_acc: 0.692000
# (Iteration 2851 / 11760) loss: 1.529869
# (Iteration 2901 / 11760) loss: 0.984167
# (Epoch 2 / 12) train acc: 0.715000; val_acc: 0.678000
# (Epoch 3 / 12) train acc: 0.717000; val_acc: 0.685000
# (Iteration 2951 / 11760) loss: 1.186100
# (Iteration 3001 / 11760) loss: 0.990401
# (Epoch 3 / 12) train acc: 0.743000; val_acc: 0.698000
# (Iteration 3051 / 11760) loss: 1.396273
# (Iteration 3101 / 11760) loss: 1.533476
# (Epoch 3 / 12) train acc: 0.735000; val_acc: 0.704000
# (Iteration 3151 / 11760) loss: 1.131978
# (Iteration 3201 / 11760) loss: 1.299430
# (Epoch 3 / 12) train acc: 0.754000; val_acc: 0.707000
# (Iteration 3251 / 11760) loss: 1.287371
# (Iteration 3301 / 11760) loss: 1.145810
# (Epoch 3 / 12) train acc: 0.784000; val_acc: 0.717000
# (Iteration 3351 / 11760) loss: 1.124301
# (Iteration 3401 / 11760) loss: 1.302360
# (Epoch 3 / 12) train acc: 0.763000; val_acc: 0.715000
# (Iteration 3451 / 11760) loss: 1.130189
# (Iteration 3501 / 11760) loss: 0.997154
# (Epoch 3 / 12) train acc: 0.757000; val_acc: 0.724000
# (Iteration 3551 / 11760) loss: 1.024031
# (Iteration 3601 / 11760) loss: 0.925049
# (Epoch 3 / 12) train acc: 0.756000; val_acc: 0.707000
# (Iteration 3651 / 11760) loss: 1.282665
# (Iteration 3701 / 11760) loss: 0.976579
# (Epoch 3 / 12) train acc: 0.775000; val_acc: 0.722000
# (Iteration 3751 / 11760) loss: 1.228804
# (Iteration 3801 / 11760) loss: 0.921672
# (Epoch 3 / 12) train acc: 0.768000; val_acc: 0.740000
# (Iteration 3851 / 11760) loss: 1.225903
# (Iteration 3901 / 11760) loss: 1.262502
# (Epoch 3 / 12) train acc: 0.777000; val_acc: 0.724000
# (Epoch 4 / 12) train acc: 0.779000; val_acc: 0.713000
# (Iteration 3951 / 11760) loss: 1.178798
# (Iteration 4001 / 11760) loss: 1.153611
# (Epoch 4 / 12) train acc: 0.775000; val_acc: 0.724000
# (Iteration 4051 / 11760) loss: 0.941662
# (Iteration 4101 / 11760) loss: 0.919599
# (Epoch 4 / 12) train acc: 0.779000; val_acc: 0.718000
# (Iteration 4151 / 11760) loss: 1.214634
# (Iteration 4201 / 11760) loss: 1.041356
# (Epoch 4 / 12) train acc: 0.782000; val_acc: 0.739000
# (Iteration 4251 / 11760) loss: 1.056904
# (Iteration 4301 / 11760) loss: 1.029168
# (Epoch 4 / 12) train acc: 0.771000; val_acc: 0.722000
# (Iteration 4351 / 11760) loss: 0.879749
# (Iteration 4401 / 11760) loss: 0.978298
# (Epoch 4 / 12) train acc: 0.807000; val_acc: 0.722000
# (Iteration 4451 / 11760) loss: 1.111619
# (Iteration 4501 / 11760) loss: 1.222298
# (Epoch 4 / 12) train acc: 0.807000; val_acc: 0.741000
# (Iteration 4551 / 11760) loss: 0.939615
# (Iteration 4601 / 11760) loss: 0.986305
# (Epoch 4 / 12) train acc: 0.804000; val_acc: 0.729000
# (Iteration 4651 / 11760) loss: 0.826382
# (Iteration 4701 / 11760) loss: 1.137678
# (Epoch 4 / 12) train acc: 0.809000; val_acc: 0.733000
# (Iteration 4751 / 11760) loss: 1.008413
# (Iteration 4801 / 11760) loss: 0.965159
# (Epoch 4 / 12) train acc: 0.810000; val_acc: 0.737000
# (Iteration 4851 / 11760) loss: 0.911961
# (Epoch 5 / 12) train acc: 0.802000; val_acc: 0.737000
# (Iteration 4901 / 11760) loss: 0.854877
# (Epoch 5 / 12) train acc: 0.816000; val_acc: 0.743000
# (Iteration 4951 / 11760) loss: 0.977991
# (Iteration 5001 / 11760) loss: 1.031379
# (Epoch 5 / 12) train acc: 0.812000; val_acc: 0.719000
# (Iteration 5051 / 11760) loss: 1.237030
# (Iteration 5101 / 11760) loss: 1.011928
# (Epoch 5 / 12) train acc: 0.793000; val_acc: 0.734000
# (Iteration 5151 / 11760) loss: 0.998533
# (Iteration 5201 / 11760) loss: 1.069028
# (Epoch 5 / 12) train acc: 0.818000; val_acc: 0.730000
# (Iteration 5251 / 11760) loss: 0.869067
# (Iteration 5301 / 11760) loss: 0.992314
# (Epoch 5 / 12) train acc: 0.819000; val_acc: 0.744000
# (Iteration 5351 / 11760) loss: 1.032274
# (Iteration 5401 / 11760) loss: 1.166080
# (Epoch 5 / 12) train acc: 0.827000; val_acc: 0.732000
# (Iteration 5451 / 11760) loss: 0.994642
# (Iteration 5501 / 11760) loss: 0.855430
# (Epoch 5 / 12) train acc: 0.839000; val_acc: 0.725000
# (Iteration 5551 / 11760) loss: 0.926237
# (Iteration 5601 / 11760) loss: 0.870623
# (Epoch 5 / 12) train acc: 0.831000; val_acc: 0.738000
# (Iteration 5651 / 11760) loss: 1.010450
# (Iteration 5701 / 11760) loss: 1.069323
# (Epoch 5 / 12) train acc: 0.811000; val_acc: 0.749000
# (Iteration 5751 / 11760) loss: 0.934299
# (Iteration 5801 / 11760) loss: 1.053869
# (Epoch 5 / 12) train acc: 0.823000; val_acc: 0.738000
# (Iteration 5851 / 11760) loss: 1.186028
# (Epoch 6 / 12) train acc: 0.845000; val_acc: 0.743000
# (Iteration 5901 / 11760) loss: 0.873892
# (Epoch 6 / 12) train acc: 0.838000; val_acc: 0.756000
# (Iteration 5951 / 11760) loss: 0.916611
# (Iteration 6001 / 11760) loss: 0.913893
# (Epoch 6 / 12) train acc: 0.840000; val_acc: 0.739000
# (Iteration 6051 / 11760) loss: 0.929447
# (Iteration 6101 / 11760) loss: 0.861042
# (Epoch 6 / 12) train acc: 0.841000; val_acc: 0.731000
# (Iteration 6151 / 11760) loss: 0.716837
# (Iteration 6201 / 11760) loss: 0.775131
# (Epoch 6 / 12) train acc: 0.852000; val_acc: 0.733000
# (Iteration 6251 / 11760) loss: 1.016587
# (Iteration 6301 / 11760) loss: 0.888849
# (Epoch 6 / 12) train acc: 0.839000; val_acc: 0.740000
# (Iteration 6351 / 11760) loss: 0.780970
# (Iteration 6401 / 11760) loss: 0.704550
# (Epoch 6 / 12) train acc: 0.868000; val_acc: 0.745000
# (Iteration 6451 / 11760) loss: 0.781423
# (Iteration 6501 / 11760) loss: 0.853088
# (Epoch 6 / 12) train acc: 0.850000; val_acc: 0.748000
# (Iteration 6551 / 11760) loss: 0.723594
# (Iteration 6601 / 11760) loss: 0.781197
# (Epoch 6 / 12) train acc: 0.867000; val_acc: 0.743000
# (Iteration 6651 / 11760) loss: 0.936393
# (Iteration 6701 / 11760) loss: 0.794133
# (Epoch 6 / 12) train acc: 0.861000; val_acc: 0.750000
# (Iteration 6751 / 11760) loss: 0.749919
# (Iteration 6801 / 11760) loss: 0.852578
# (Epoch 6 / 12) train acc: 0.881000; val_acc: 0.749000
# (Iteration 6851 / 11760) loss: 0.805281
# (Epoch 7 / 12) train acc: 0.865000; val_acc: 0.747000
# (Iteration 6901 / 11760) loss: 1.045644
# (Epoch 7 / 12) train acc: 0.874000; val_acc: 0.754000
# (Iteration 6951 / 11760) loss: 0.965821
# (Iteration 7001 / 11760) loss: 1.047465
# (Epoch 7 / 12) train acc: 0.866000; val_acc: 0.768000
# (Iteration 7051 / 11760) loss: 0.731862
# (Iteration 7101 / 11760) loss: 0.882851
# (Epoch 7 / 12) train acc: 0.869000; val_acc: 0.754000
# (Iteration 7151 / 11760) loss: 0.886734
# (Iteration 7201 / 11760) loss: 1.040747
# (Epoch 7 / 12) train acc: 0.867000; val_acc: 0.764000
# (Iteration 7251 / 11760) loss: 0.885465
# (Iteration 7301 / 11760) loss: 0.865805
# (Epoch 7 / 12) train acc: 0.875000; val_acc: 0.753000
# (Iteration 7351 / 11760) loss: 0.924064
# (Iteration 7401 / 11760) loss: 0.708725
# (Epoch 7 / 12) train acc: 0.874000; val_acc: 0.749000
# (Iteration 7451 / 11760) loss: 0.918623
# (Iteration 7501 / 11760) loss: 0.786888
# (Epoch 7 / 12) train acc: 0.870000; val_acc: 0.771000
# (Iteration 7551 / 11760) loss: 0.743911
# (Iteration 7601 / 11760) loss: 0.613105
# (Epoch 7 / 12) train acc: 0.890000; val_acc: 0.755000
# (Iteration 7651 / 11760) loss: 0.694688
# (Iteration 7701 / 11760) loss: 0.890472
# (Epoch 7 / 12) train acc: 0.849000; val_acc: 0.747000
# (Iteration 7751 / 11760) loss: 0.731369
# (Iteration 7801 / 11760) loss: 0.726302
# (Epoch 7 / 12) train acc: 0.880000; val_acc: 0.766000
# (Epoch 8 / 12) train acc: 0.891000; val_acc: 0.758000
# (Iteration 7851 / 11760) loss: 0.659098
# (Iteration 7901 / 11760) loss: 0.945286
# (Epoch 8 / 12) train acc: 0.893000; val_acc: 0.759000
# (Iteration 7951 / 11760) loss: 1.004577
# (Iteration 8001 / 11760) loss: 0.745491
# (Epoch 8 / 12) train acc: 0.881000; val_acc: 0.761000
# (Iteration 8051 / 11760) loss: 0.723371
# (Iteration 8101 / 11760) loss: 0.897876
# (Epoch 8 / 12) train acc: 0.878000; val_acc: 0.749000
# (Iteration 8151 / 11760) loss: 0.835979
# (Iteration 8201 / 11760) loss: 0.633891
# (Epoch 8 / 12) train acc: 0.893000; val_acc: 0.768000
# (Iteration 8251 / 11760) loss: 0.723911
# (Iteration 8301 / 11760) loss: 0.661336
# (Epoch 8 / 12) train acc: 0.886000; val_acc: 0.767000
# (Iteration 8351 / 11760) loss: 1.162876
# (Iteration 8401 / 11760) loss: 0.818023
# (Epoch 8 / 12) train acc: 0.893000; val_acc: 0.745000
# (Iteration 8451 / 11760) loss: 0.611116
# (Iteration 8501 / 11760) loss: 0.612532
# (Epoch 8 / 12) train acc: 0.894000; val_acc: 0.763000
# (Iteration 8551 / 11760) loss: 0.766848
# (Iteration 8601 / 11760) loss: 0.757009
# (Epoch 8 / 12) train acc: 0.893000; val_acc: 0.746000
# (Iteration 8651 / 11760) loss: 0.726760
# (Iteration 8701 / 11760) loss: 0.812795
# (Epoch 8 / 12) train acc: 0.904000; val_acc: 0.747000
# (Iteration 8751 / 11760) loss: 0.583004
# (Iteration 8801 / 11760) loss: 0.806673
# (Epoch 8 / 12) train acc: 0.899000; val_acc: 0.765000
# (Epoch 9 / 12) train acc: 0.908000; val_acc: 0.752000
# (Iteration 8851 / 11760) loss: 0.742782
# (Iteration 8901 / 11760) loss: 0.944126
# (Epoch 9 / 12) train acc: 0.869000; val_acc: 0.752000
# (Iteration 8951 / 11760) loss: 0.686754
# (Iteration 9001 / 11760) loss: 0.981476
# (Epoch 9 / 12) train acc: 0.907000; val_acc: 0.749000
# (Iteration 9051 / 11760) loss: 0.866599
# (Iteration 9101 / 11760) loss: 0.918607
# (Epoch 9 / 12) train acc: 0.895000; val_acc: 0.760000
# (Iteration 9151 / 11760) loss: 0.579047
# (Iteration 9201 / 11760) loss: 0.796029
# (Epoch 9 / 12) train acc: 0.897000; val_acc: 0.758000
# (Iteration 9251 / 11760) loss: 0.609972
# (Iteration 9301 / 11760) loss: 0.725903
# (Epoch 9 / 12) train acc: 0.900000; val_acc: 0.744000
# (Iteration 9351 / 11760) loss: 0.808947
# (Iteration 9401 / 11760) loss: 0.751447
# (Epoch 9 / 12) train acc: 0.914000; val_acc: 0.755000
# (Iteration 9451 / 11760) loss: 0.889590
# (Iteration 9501 / 11760) loss: 0.655373
# (Epoch 9 / 12) train acc: 0.930000; val_acc: 0.751000
# (Iteration 9551 / 11760) loss: 0.516689
# (Iteration 9601 / 11760) loss: 0.508476
# (Epoch 9 / 12) train acc: 0.910000; val_acc: 0.762000
# (Iteration 9651 / 11760) loss: 0.534051
# (Iteration 9701 / 11760) loss: 0.636081
# (Epoch 9 / 12) train acc: 0.916000; val_acc: 0.765000
# (Iteration 9751 / 11760) loss: 0.591601
# (Epoch 10 / 12) train acc: 0.911000; val_acc: 0.751000
# (Iteration 9801 / 11760) loss: 1.043490
# (Epoch 10 / 12) train acc: 0.884000; val_acc: 0.751000
# (Iteration 9851 / 11760) loss: 0.615729
# (Iteration 9901 / 11760) loss: 0.693707
# (Epoch 10 / 12) train acc: 0.913000; val_acc: 0.764000
# (Iteration 9951 / 11760) loss: 0.595777
# (Iteration 10001 / 11760) loss: 0.667738
# (Epoch 10 / 12) train acc: 0.909000; val_acc: 0.785000
# (Iteration 10051 / 11760) loss: 0.582832
# (Iteration 10101 / 11760) loss: 0.719282
# (Epoch 10 / 12) train acc: 0.912000; val_acc: 0.786000
# (Iteration 10151 / 11760) loss: 0.657505
# (Iteration 10201 / 11760) loss: 0.810533
# (Epoch 10 / 12) train acc: 0.908000; val_acc: 0.768000
# (Iteration 10251 / 11760) loss: 0.724561
# (Iteration 10301 / 11760) loss: 0.572427
# (Epoch 10 / 12) train acc: 0.917000; val_acc: 0.756000
# (Iteration 10351 / 11760) loss: 0.573227
# (Iteration 10401 / 11760) loss: 0.811961
# (Epoch 10 / 12) train acc: 0.909000; val_acc: 0.750000
# (Iteration 10451 / 11760) loss: 0.608629
# (Iteration 10501 / 11760) loss: 0.731176
# (Epoch 10 / 12) train acc: 0.920000; val_acc: 0.767000
# (Iteration 10551 / 11760) loss: 0.555459
# (Iteration 10601 / 11760) loss: 0.579929
# (Epoch 10 / 12) train acc: 0.924000; val_acc: 0.767000
# (Iteration 10651 / 11760) loss: 0.693260
# (Iteration 10701 / 11760) loss: 0.681482
# (Epoch 10 / 12) train acc: 0.916000; val_acc: 0.759000
# (Iteration 10751 / 11760) loss: 0.620229
# (Epoch 11 / 12) train acc: 0.922000; val_acc: 0.754000
# (Iteration 10801 / 11760) loss: 0.612244
# (Epoch 11 / 12) train acc: 0.905000; val_acc: 0.756000
# (Iteration 10851 / 11760) loss: 0.485354
# (Iteration 10901 / 11760) loss: 0.666904
# (Epoch 11 / 12) train acc: 0.935000; val_acc: 0.774000
# (Iteration 10951 / 11760) loss: 0.612741
# (Iteration 11001 / 11760) loss: 0.577029
# (Epoch 11 / 12) train acc: 0.945000; val_acc: 0.764000
# (Iteration 11051 / 11760) loss: 0.609063
# (Iteration 11101 / 11760) loss: 0.803309
# (Epoch 11 / 12) train acc: 0.933000; val_acc: 0.770000
# (Iteration 11151 / 11760) loss: 0.491234
# (Iteration 11201 / 11760) loss: 0.507729
# (Epoch 11 / 12) train acc: 0.918000; val_acc: 0.771000
# (Iteration 11251 / 11760) loss: 0.643952
# (Iteration 11301 / 11760) loss: 0.572661
# (Epoch 11 / 12) train acc: 0.928000; val_acc: 0.779000
# (Iteration 11351 / 11760) loss: 0.497848
# (Iteration 11401 / 11760) loss: 0.648052
# (Epoch 11 / 12) train acc: 0.918000; val_acc: 0.768000
# (Iteration 11451 / 11760) loss: 0.500725
# (Iteration 11501 / 11760) loss: 0.671980
# (Epoch 11 / 12) train acc: 0.918000; val_acc: 0.777000
# (Iteration 11551 / 11760) loss: 0.647050
# (Iteration 11601 / 11760) loss: 0.483820
# (Epoch 11 / 12) train acc: 0.938000; val_acc: 0.777000
# (Iteration 11651 / 11760) loss: 0.592151
# (Iteration 11701 / 11760) loss: 0.549258
# (Epoch 11 / 12) train acc: 0.927000; val_acc: 0.778000
# (Iteration 11751 / 11760) loss: 0.506156
# (Epoch 12 / 12) train acc: 0.933000; val_acc: 0.774000
# Final training