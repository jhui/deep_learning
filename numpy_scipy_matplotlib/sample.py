import numpy as np
import matplotlib as plt

size = 100
sigma_x = 6.
sigma_y = 2.

x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)

x, y = np.meshgrid(x, y)
z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
     + y**2/(2*sigma_y**2))))

plt.contourf(x, y, z, cmap='Blues')
plt.colorbar()
plt.show()