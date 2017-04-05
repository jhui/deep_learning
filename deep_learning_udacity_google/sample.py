import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 1
loop = 200_000

v = np.random.normal(mu, sigma, loop)
print(np.var(v))
plt.hist(v, bins=200, normed=1)
plt.xlim([-70, 70])

plt.show()

l = []
for i in range(loop):
    v = np.random.normal(mu, 2/np.sqrt(500), 500)
    sum = np.sum(v)
    l.append(sum)
print(np.var(l))

plt.hist(l, bins=200, normed=1)
plt.xlim([-70, 70])

plt.show()

