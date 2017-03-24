import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def eye(N):
    noise = [ [i, i + 0.3 * np.random.randn(1)[0]] for i in range(N)]
    return noise

N = 20
a = eye(N)
b = a[0::2]

print(a)
print(b)