import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA

iteration = 100_000
learning_rate = 1e-4
N = 100
LAYER = 4
LAST = LAYER - 1

def true_y(education, income):
    dates = 0.8 * education + 0.2 * income + 2
    return dates

def sample(education, income, verbose=False):
    dates = true_y(education, income)
    noise =  dates * 0.15 * np.random.randn(education.shape[0]) # Add some noise2c_date.py
    if verbose:
        print(dates)
        print(noise)
    return dates + noise

def affine_forward(x, W, b):
    # x: input sample (N, D)
    # W: Weight (2, k)
    # b: bias (k,)
    # out: (N, k)
    out = x.dot(W) + b           # (N, D) * (D, k) -> (N, k)
    cache = (x, W, b)
    return out, cache

def affine_backward(dout, cache):
    # dout: dJ/dout (N, K)
    # x: input sample (N, D)
    # W: Weight (D, K)
    # b: bias (K, )
    x, W, b = cache

    if len(dout.shape)==1:
        dout = dout[:, np.newaxis]
    if len(W.shape) == 1:
        W = W[:, np.newaxis]

    W = W.T
    dx = dout.dot(W)

    dW = x.T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dW, db

def relu_forward(x):
    cache = x
    out = np.maximum(0, x)
    return out, cache

def relu_backward(dout, cache):
    out = cache
    dh = dout
    dh[out < 0] = 0
    return dh

def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    return out, out

def sigmoid_backward(dout, cache):
    dh = cache * (1-cache)
    return dh * dout

def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    N = X.shape[0]                        # Find the number of samples
    h = h.reshape(-1)
    loss = np.sum(np.square(h - y)) / N   # Compute the mean square error from its true value y
    nh = h.reshape(-1)
    dout = 2 * (nh-y) / N                  # Compute the partial derivative of J relative to out
    return loss, dout

def compute_loss(X, W, b, y=None, use_relu=True):
    cache_z = [None] * LAYER
    cache_relu = [None] * (LAST)

    h = X

    for i in range(LAST):
        h, cache_z[i] = affine_forward(h, W[i], b[i])
        h, cache_relu[i] = relu_forward(h)

    h, cache_z[LAST] = affine_forward(h, W[LAST], b[LAST])

    if y is None:
        return h, None, None

    dW = [None] * LAYER
    db = [None] * LAYER
    loss, dout = mean_square_loss(h, y)

    dz, dW[LAST], db[LAST] = affine_backward(dout, cache_z[LAST])

    for i in reversed(range(LAST)):
        dh = relu_backward(dz, cache_relu[i])
        dz, dW[i], db[i] = affine_backward(dh, cache_z[i])

    return loss, dW, db


education = np.random.randint(26, size=N) # (N,) Generate 10 random sample with years of education from 0 to 25.
income = np.random.randint(100, size=education.shape[0]) # (N,) Generate the corresponding income.

# The number of dates according to the formula of an Oracle.
# In practice, the value come with each sample data.
Y = sample(education, income)    # (N,)

W = [None] * LAYER
b = [None] * LAYER

W[0] = np.array([[0.7, 0.05, 0.0, 0.2, 0.2], [0.3, 0.01, 0.01, 0.3, 0.2]])  # (2, K)
b[0] = np.array([0.8, 0.2, 1.0, 0.2, 0.1])

for i in range(1, LAST):
    W[i] = np.array([[0.7, 0.05, 0.0, 0.2, 0.1], [0.3, 0.01, 0.01, 0.3, 0.1], [0.3, 0.01, 0.01, 0.3, 0.1], [0.07, 0.04, 0.01, 0.1, 0.1], [0.1, 0.1, 0.01, 0.02, 0.03]])  # (K, K)
    b[i] = np.array([0.8, 0.2, 1.0, 0.2, 0.1])

W[LAST] = np.array([[0.8], [0.5], [0.05], [0.05], [0.1]])                 # (K, 1)
b[LAST] = np.array([0.2])

X = np.concatenate((education[:, np.newaxis], income[:, np.newaxis]), axis=1) # (N, 2) N samples with 2 features

for i in range(iteration):
    loss, dW, db = compute_loss(X, W, b, Y)
    for j, (cdW, cdb) in enumerate(zip(dW, db)):
        W[j] -= learning_rate * cdW
        b[j] -= learning_rate * cdb
    if i%20000==0:
        print(f"iteration {i}: loss={loss:.4}")

print(f"W = {W}")
print(f"b = {b}")

TN = 100
test_education = np.full(TN, 22)
test_income = np.random.randint(TN, size=test_education.shape[0])
test_income = np.sort(test_income)

true_model_Y = true_y(test_education, test_income)
true_sample_Y = sample(test_education, test_income, verbose=False)
X = np.concatenate((test_education[:, np.newaxis], test_income[:, np.newaxis]), axis=1)

out, _, _ = compute_loss(X, W, b)
loss_model, _ = mean_square_loss(out, true_model_Y)
loss_sample, _ = mean_square_loss(out, true_sample_Y)

print(f"testing: loss (compare with Oracle)={loss_model:.6}")
print(f"testing: loss (compare with sample)={loss_sample:.4}")

plt.colors()
plt.scatter(test_income, true_model_Y, alpha=0.4)

plt.plot(test_income, out, color="r")
plt.legend(['prediction', 'true label'])

plt.show()


plot_education = np.arange(0, 25, 0.25)
plot_income = np.arange(0, 100, 1)

len_edu = len(plot_education)
len_income = len(plot_income)

data = np.array([[x * 0.25 , y] for x in plot_education for y in plot_income])
out, _, _ = compute_loss(data, W, b)
Z = out.reshape(len_edu, len_income)

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(plot_education, plot_income)

colortuple = ('y', 'b')
colors = np.empty(X.shape, dtype=str)
for y in range(len_income):
    for x in range(len_edu):
        colors[x, y] = colortuple[(x + y) % len(colortuple)]

surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)

ax.w_zaxis.set_major_locator(LinearLocator(6))

plt.show()

