import numpy as np
import matplotlib.pyplot as pl

# Noiseless training dataset
Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
Xtrain = np.array([-4, -3]).reshape(2,1)
ytrain = np.sin(Xtrain)

# 50 Test data linear distributed between -5 and 5.
n = 50
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# A kernel function (aka Gaussian) measuring the similarity between a and b. 1 means the same.
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

param = 0.1
K_ss = kernel(Xtest, Xtest, param)    # Shape (50, 50)

# Get cholesky decomposition (square root) of the covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))

# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(n,3)))

# Now let's plot the 3 sampled functions.
pl.plot(Xtest, f_prior)
pl.axis([-5, 5, -3, 3])
pl.title('Three samples from the GP prior')
pl.show()

# Apply the kernel function to our training points
K = kernel(Xtrain, Xtrain, param)                        # Shape (5, 5)
K_s = kernel(Xtrain, Xtest, param)                       # Shape (5, 50)

L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))  # Shape (5, 5)

# Compute the mean at our test points.
Lk = np.linalg.solve(L, K_s)                             # Shape (5, 50)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,)) # Shape (50, )

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)               # Shape (50, )
stdv = np.sqrt(s2)                                       # Shape (50, )

# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))    # Shape (50, 50)
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,5))) # Shape (50, 3)

pl.plot(Xtrain, ytrain, 'bs', ms=8)
pl.plot(Xtest, f_post)
pl.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.axis([-5, 5, -3, 3])
pl.title('Three samples from the GP posterior')
pl.show()