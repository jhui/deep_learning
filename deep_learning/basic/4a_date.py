import numpy as np
import matplotlib.pyplot as plt

iteration = 100000
learning_rate = 1e-5
N = 100

def true_y(education, income):
    dates = 0.8 * education + 0.3 * income + 2
    return dates

def sample(education, income):
    dates = true_y(education, income)
    dates += dates * 0.01 * np.random.randn(education.shape[0]) # Add some noise2c_date.py
    return dates

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

def mean_square_loss(h, y):
    # h: prediction (N,)
    # y: true value (N,)
    N = X.shape[0]                        # Find the number of samples
    h = h.reshape(-1)
    loss = np.sum(np.square(h - y)) / N   # Compute the mean square error from its true value y
    nh = h.reshape(-1)
    dout = 2 * (nh-y) / N                  # Compute the partial derivative of J relative to out
    return loss, dout


def compute_loss(X, W, b, y=None):
    z1, cache_z1 = affine_forward(X, W[0], b[0])
    h1, cache_relu1 = relu_forward(z1)
    z2, cache_z2 = affine_forward(h1, W[1], b[1])

    if y is None:
        return z2, None, None

    dW = [None] * 2
    db = [None] * 2
    loss, dout = mean_square_loss(z2, y)
    dh2, dW[1], db[1] = affine_backward(dout, cache_z2)

    dh1 = relu_backward(dh2, cache_relu1)
    _, dW[0], db[0] = affine_backward(dh1, cache_z1)

    return loss, dW, db


education = np.random.randint(26, size=N) # (N,) Generate 10 random sample with years of education from 0 to 25.
income = np.random.randint(100, size=education.shape[0]) # (N,) Generate the corresponding income.

# The number of dates according to the formula of an Oracle.
# In practice, the value come with each sample data.
true_model_Y = sample(education, income)    # (N,)

W = [None] * 2
b = [None] * 2

W[0] = np.array([[0.7, 0.05, 0.0, 0.2], [0.3, 0.01, 0.01, 0.3]])  # (2, K)
b[0] = np.array([0.8, 0.2, 1.0, 0.2])

W[1] = np.array([[0.8], [0.5], [0.05], [0.05]])                 # (K, 1)
b[1] = np.array([0.2])

X = np.concatenate((education[:, np.newaxis], income[:, np.newaxis]), axis=1) # (N, 2) N samples with 2 features

for i in range(iteration):
    loss_model, dW, db = compute_loss(X, W, b, true_model_Y)
    for j, (cdW, cdb) in enumerate(zip(dW, db)):
        W[j] -= learning_rate * cdW
        b[j] -= learning_rate * cdb
    if i%1000==0:
        print(f"iteration {i}: loss={loss_model:.4}")

print(f"W = {W}")
print(f"b = {b}")

TN = 100
test_education = np.full(TN, 22)
test_income = np.random.randint(TN, size=test_education.shape[0])
test_income = np.sort(test_income)

true_model_Y = true_y(test_education, test_income)
true_sample_Y = sample(test_education, test_income)
X = np.concatenate((test_education[:, np.newaxis], test_income[:, np.newaxis]), axis=1)

out, _, _ = compute_loss(X, W, b)
loss_model, _ = mean_square_loss(out, true_model_Y)
loss_sample, _ = mean_square_loss(out, true_sample_Y)

print(f"testing: loss (compare with Oracle)={loss_model:.6}")
print(f"testing: loss (compare with sample)={loss_sample:.2}")

plt.colors()
plt.scatter(test_income, true_model_Y, alpha=0.4)
plt.scatter(test_income, true_sample_Y, alpha=0.4)

plt.plot(test_income, out, color="r")
plt.legend(['prediction', 'true label', 'sample label'])

plt.show()

