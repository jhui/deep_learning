import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = np.array([[0., -0.1], [1.7, .4]])

x1 = np.dot(np.random.randn(100, 2), cov)
x2 = np.random.multivariate_normal(mean, cov.T.dot(cov), 100)

plt.plot(x1[:, 0], x1[:, 1], 'g*', alpha=0.2)
plt.plot(x2[:, 0], x2[:, 1], 'ko', alpha=0.2, ms=4)

plt.show()
