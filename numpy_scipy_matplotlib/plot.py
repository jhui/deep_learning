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

# Plot 6
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
xlen = len(X)
Y = np.arange(-5, 5, 0.25)
ylen = len(Y)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Create an empty array of strings with the same shape as the meshgrid, and
# populate it with two colors in a checkerboard pattern.
colortuple = ('y', 'b')
colors = np.empty(X.shape, dtype=str)
for y in range(ylen):
    for x in range(xlen):
        colors[x, y] = colortuple[(x + y) % len(colortuple)]

# Plot the surface with face colors taken from the array we made.
surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)

# Customize the z axis.
ax.set_zlim(-1, 1)
ax.w_zaxis.set_major_locator(LinearLocator(6))

plt.show()

# Plot 7
x = np.arange(0, 10 * np.pi, 1)
y = np.sin(x)
plt.scatter(x, y)
plt.plot(x, y)
plt.annotate(f"({x} {y})",
             xy=(x, y),
             xytext=(5, 2),
             textcoords='offset points',
             ha='right',
             va='bottom')
plt.show()

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
