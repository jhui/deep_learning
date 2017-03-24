import numpy as np
import matplotlib.pyplot as plt

a=[[0, 0], [2, 1.0], [4, 4.8], [6, 6.3], [8, 7.6], [10, 10], [12, 11.6], [14, 12.8], [16, 17.2], [18, 18.4], [20, 20]]
plt.plot(*zip(*a), marker='o', color='r', ls='')

x = np.arange(0, 20.3, 0.1)
y = 1.92714372    * 10 ** -7 * x ** 9 \
    - 1.613583798 * 10 ** -5 * x ** 8 \
    + 5.569160694 * 10 ** -4 * x ** 7 \
    - 1.021370538 * 10 ** -2 * x ** 6 \
    + 1.067156383 * 10 ** -1 * x ** 5 \
    - 6.286375234 * 10 ** -1 * x ** 4 \
    + 1.900256362 * 10 ** 0 * x ** 3 \
    - 2.18932239  * 10 ** 0 * x ** 2 \
    + 8.960617855 * 10 ** -1 * x \
    - 8.220605079 * 10 **-3

plt.colors()
plt.plot(x, y)
plt.plot(x, x)
plt.show()

plt.colors()
plt.plot(*zip(*a), marker='o', color='r', ls='')
plt.plot(x, x)
plt.show()

plt.plot(*zip(*a), marker='o', color='r', ls='')
plt.show()
