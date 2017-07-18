import numpy as np
import matplotlib as plt

v = np.histogram([0.5, 1.5, 2, 1, 3, 1.9], bins=[0, 1, 2, 3])
print(v)