########## Plotting
import numpy as np
import matplotlib.pyplot as plt

# Plot 1
mu, sigma = 1, 0.65
v = np.random.normal(mu, sigma, 20000)
plt.hist(v, bins=100, normed=1)
plt.show()

# Plot 2
(n, bins) = np.histogram(v, bins=100, normed=True)
plt.plot(.5 * (bins[1:] + bins[:-1]), n)
plt.show()

# Plot 3
x = np.arange(0, 10 * np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# Plot 4
x = np.arange(0, 10 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.axhline(y=0, color=0.8)     # horizontal axis
plt.show()

# Plot 5
x = np.arange(0, 10 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()

# plt.ylim(bottom=0)   # set y bottom limit to 0

########## images
from scipy.misc import imread, imsave, imresize

img = imread('a.png')
img.dtype                       # uint8
img.shape                       # (28, 28)

img_tinted = img * [0.95]       # [0.9, 0,95, 0.85] for RGB images
img_tinted = imresize(img_tinted, (300, 300))
imsave('a_tinted.png', img_tinted)


img = imread('a.png')
img_tinted = img * [0.95]

plt.subplot(1, 2, 1)             # (1x2) plot. subplot index start from 1
plt.imshow(img)

plt.subplot(1, 2, 2)             # index 2
# Cast the image to uint8 before display it to avoid strange behavior.
plt.imshow(np.uint8(img_tinted))
plt.show()
