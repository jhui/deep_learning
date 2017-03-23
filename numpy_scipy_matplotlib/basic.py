import numpy as np

########### Meta data
a = np.arange(10).reshape(2, 5)     # ndarray([[ 0  1  2  3  4],
                                    #          [ 5  6  7  8  9]]
print(a)
a.shape             # (2, 5)
a.ndim              # 2
a.dtype.name        # 'int64' (others: float32, float64,...)
a.itemsize          # 8
a.size              # 10
type(a)             # <type 'numpy.ndarray'>

########### Creation
b = np.array([1, 2, 3])
b = np.array([[1.1, 2.0, 3], [4, 5, 6]])
b = np.array([(1.1, 2.0, 3), (4, 5, 6)])

b = np.array([[1.1, 2, 3], [4, 5, 6]], dtype=np.float32)  # Set type

np.empty((2, 4))                    # Uninitialized. Value un-determine.
np.zeros((2, 4))
np.ones((2, 4, 5), dtype=np.int16)
np.ones_like(b)                     # Create an array with shape like b. ndarray([[ 1.  1.  1.], [ 1.  1.  1.]])
np.full((2, 4), 5)                  # Fill with 5
np.eye(2)                           # Identity matrix

np.arange(6)                        # [0 1 2 3 4 5]
np.arange(10, 25, 5)                # ndarray([10 15 20])
np.arange(0, 1, 0.2)                # ndarray([ 0. 0.2  0.4  0.6  0.8]

np.linspace(0, 2, 8)                # 9 numbers (0 to 2 inclusive) ndarray([0. 0.25 0.5 ... 1.75 2.])
np.logspace(1.0, 2.0, num=4)        # ndarray([10. 21.5443469 46.41588834  100.]

np.random.random((2, 4))            # [0, 1)

np.random.randint(5, size=10)       # number from 0 to 4
np.random.randint(5, size=(2, 4))

np.random.randn(10, 2, 3)           # 10x2x3 array with normal distribution

a = np.array([1, 5, 7, 8])
c = np.random.choice(a, 3, replace=False)   # ndarray([8 5 7])

c = np.random.choice(len(a), 2)
a[c]

def f(x, y):
    return 10 * x + y

b = np.fromfunction(f, (5, 4),
                    dtype=np.int)  # ndarray([[ 0,  1,  2,  3], [10, 11, 12, 13], [20, 21, 22, 23], ..., [40, 41, 42, 43]])

# Creation
# arange, array, copy, empty, empty_like, eye, fromfile, fromfunction,
# identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like

########### Reshape, Shape manipulation
np.arange(12).reshape(4, 3)         # [[ 0  1  2] [ 3  4  5] [ 6  7  8] [ 9 10 11]]
np.arange(24).reshape(2, 3, 4)

a = np.arange(20)
a.shape = 2, -1, 2                  # -1 means "whatever it should be"
a.shape                             # (2, 5, 2)

a = np.arange(12).reshape(4, 3)
a.ravel()                           # flattened ndarray([ 0 1 2 ... 11])
a.reshape(6, 2)                     # ndarray([[ 0 1]  [2 3] ...[10 11]])
a.reshape(3, -1)                    # -1 will automatic calculated as 4
a.T                                 # transposed ndarray([[ 0 3 6 9] [1 4 7 10] [2 5 8 11]])
a.T.shape                           # (3, 4)

a = np.arange(6)                    # ndarray([0, 1, 2, 3, 4, 5])
a.reshape(2, 3)                     # a remains as ndarray [0 1 2 3 4 5]
a.resize((2, 3))                    # a changes to [[1 2 3] [ 4 5 6]]

a = np.array([4., 2.])
v = a[:, np.newaxis]                # ndarray([[ 4.], [ 2.]])
v = a[np.newaxis, :]                # ndarray([[ 4.  2.]])

a = np.arange(6)                    # ndarray([0, 1, 2, 3, 4, 5])
np.reshape(a, (a.shape[0], -1))     # a[:, np.newaxis]

C = 3
x = np.random.randn((100, C, 10, 10))
x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)

########### Indexing
# Access
a = np.arange(10)
a[2]                                # 2
b = a[2:6]                          # ndarray([2 3 4 5])
a[:6:2] = 0                         # equivalent to a[0:6:2] = 0. 0 to 6 exclusive, set every other element to 0
b = a[::-1]                         # reversed an array

a = np.arange(12).reshape(3, 4)     # ndarray([[ 0  1  2  3]
                                    #          [ 4  5  6  7]
                                    #          [ 8  9 10 11]]

a[0:2]                              # ndarray([[ 0  1  2  3]
                                    #          [ 4  5  6  7]
a[2, 3]                             # 11
a[0:, 1]                            # Second column of each row ndarray([ 1 5 9])
a[:, 1]                             # same
a[1:3, :]                           # Row 1 and 2

a[-1]                               # The last row. Equivalent to b[-1,:]

a = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[10, 11, 12],
               [13, 14, 15]]])
for row in a:
    pass  # each row
for element in a.flat:
    pass  # each element

a.shape                             # (2, 2, 3)
a[1, ...]                           # a[1,:,:] or a[1] [[10 11 12] [13 14 15]]
a[..., 2]                           # same as a[:,:,2] [[ 2 5] [12 15]]

# Assign
a = np.arange(5)                    # ndarray([0 1 2 3 4])
a[[1, 2, 4]] = 0                    # ndarray([0 0 0 3 0])

a = np.arange(5)                    # ndarray([0 1 2 3 4])
a[[0, 0, 2]] = [1, 2, 3]            # ndarray([2 1 3 3 4])

a = np.arange(12).reshape(3, 4)
a[[0, 2]] = 0                       # ndarray([[0 0 0 0]
                                    #          [4 5 6 7]
                                    #          [0 0 0 0]])

y = np.array([1, 2, 4, 2, 1])       # Find index with element value==2
index = np.flatnonzero(y == 2)      # ndarray([1 3])

# Mask by boolean
a = np.arange(12).reshape(3, 4)
b = a > 4                           # ndarray([[False, False, False, False],
                                    #          [False,  True,  True,  True],
                                    #          [True,  True,  True,  True]], dtype=bool)

a[b]                                # 1d array ndarray([ 5,  6,  7,  8,  9, 10, 11])

a[a>4] = 0                          # All elements of 'a' higher than 4 become 0
                                    # ndarray([[0, 1, 2, 3]
                                    #          [4, 0, 0, 0]
                                    #          [0, 0, 0, 0]])


a = np.arange(12).reshape(3, 4)
b1 = np.array([False, True, True])          # 1D selection
b2 = np.array([True, False, True, True])    # 2D selection
a[b1, :]                            # selecting rows
                                    # ndarray([[ 4,  5,  6,  7]
                                    #          [ 8,  9, 10, 11]])
a[b1]                               # same

b = a[:, b2]                        # selecting columns
                                    # ndarray([[ 0 2 3]
                                    #          [ 4 6 7]
                                    #          [ 8 10 11]])

a = np.arange(12).reshape(3, 4)     # ndarray([[ 0  1  2  3]
                                    #          [ 4  5  6  7]
                                    #          [ 8  9 10 11]])
y = np.array([0, 3, 1])
a[range(a.shape[0]), y]             # ndarray([0 7 9])
a[:, y]                             # Not the same as above
                                    # ndarray([[ 0  3  1]
                                    #          [ 4  7  5]
                                    #          [ 8 11  9]]

x = np.array([0, 2, 1])
y = np.array([0, 3, 1])             # Dimension of x and y needs to be match.
b = a[x, y]                         # ndarray([0 11 5])


########## Mask by index
a = np.arange(12) * 2               # ndarray([ 0  2  4  6  8 10 12 14 16 18 20 22])

# The return result will have the same shape as the index.
# Each index will be replaced by the element it indexed.
i = np.array([1, 1, 4, 8, 2])       # an array of indices
a[i]                                # ndarray([2 2 8 16 4])

j = np.array([[3, 4], [5, 1]])
a[j]                                # the same shape as j [[ 6  8]
                                    #                      [10 2]]
palette = np.array([[0, 0, 0],        # black
                    [255, 255, 255],  # white
                    [255, 0, 0],      # red
                    [0, 255, 0],      # green
                    [0, 0, 255]])     # blue
image = np.array([[0, 1, 3, 0],       # each value index to the palette above
                  [0, 2, 4, 0]])

palette[image]                        # Replace each index with its value
                                      # [[[  0   0   0]
                                      #   [255 255 255]
                                      #   [  0 255   0]
                                      #   [  0   0   0]]
                                      #  [[  0   0   0]
                                      #   [255   0   0]
                                      #   [  0   0 255]
                                      #   [  0   0   0]]]

### For multi-dimensional indexing
a = np.arange(12).reshape(3, 4)  # ndarray([[ 0,  1,  2,  3],
                                 #          [ 4,  5,  6,  7],
                                 #          [ 8,  9, 10, 11]])
i = np.array([[0, 1],
              [1, 2]])
j = np.array([[2, 1],
              [3, 0]])

# i, j must be same shape (or broadcast) so they can combined to form a 2D index
a[i, j]                          # [[2 5]
                                 #  [7 8]]
                                 # [[a[0, 2] a[1, 1]
                                 #  [a[1, 3] a[2, 0]]

a[i, 2]                          # ndarray([[ 2  6]
                                 #          [ 6 10]])

b = a[:, j]
                                 # [[[ 2  1]
                                 #   [ 3  0]]
                                 #  [[ 6  5]
                                 #   [ 7  4]]
                                 #  [[10  9]
                                 #   [11  8]]]

l = [i, j]
a[l]                             # a[i,j]

a = np.array([[1, 4, 6], [3, 2, 4]])
max_column_index = a.argmax(axis=0)                # ndarray([1 0 0 ])
max_data = a[max_column_index, range(a.shape[1])]  # ndarray([3 4 6])
np.all( max_data == a.max(axis=0))                 # True


########### Broadcasting
# A      (4d array):  8 x 1 x 6 x 1
# B      (3d array):      7 x 1 x 5
# Result (4d array):  8 x 7 x 6 x 5

x = np.arange(2)
X1 = x.reshape(2, 1)

y = np.ones(3)

z = np.ones((3, 2))

x.shape                # (2,)
y.shape                # (3,)
X1.shape               # (2, 1)

# x + y          # ValueError: operands could not be broadcast together with shapes (2,) (3,)

(X1 + y).shape   # (2, 3)

X1 + y           # ndarray([[ 1 1 1]
                 #          [ 2 2 2]])

(x + z).shape    # (3, 2)

x + z            # ndarray([[1 2]
                 #          [1 2]
                 #          [1 2]])

a = np.array([0.0, 10.0, 20.0])
b = np.array([1.0, 2.0])
v = a[:, np.newaxis] + b   # ndarray([[ 1 2]
                           #          [ 11 12]
                           #          [ 21 22]]

########### Operation
a = np.array([2.0, 3.1, 6, 7])
b = np.arange(4)

b ** 2
10 * np.cos(a)
a < 30                           # ndarray([ True, True, False, False], dtype=bool)

c = a - b

a *= 4
a += b                           # Automatic up-casting b to float

A = np.array([[4, 1], [5, 1]])
B = np.array([[2, 1], [1, 2]])

A * B                            # element-wise product

v = A.dot(B)                     # matrix product ndarray([[9 6] [11  7]])
np.dot(A, B)                     # same

a1 = np.array([[1, 2, 3]])       # (1, 3)
a2 = np.array([4, 5, 6])         # (3, )
a3 = np.array([4])               # (1, )
a4 = np.array([[4], [5], [6]])   # (3, 1)


v1 = a1 + a3                     # (1, 3)
v2 = a1.dot(a2)                  # (1,)
v3 = a1.dot(a4)                  # (1, 1)

########### Conditional
x = np.random.randn(100)
dout = np.random.randn(100)
dx = np.where(x > 0, dout, 0)

########### Aggregiation
a.sum()
a.min()
a.max()

a = np.array([1, 1, 3, 3, 1])
b = np.array([1, 4, 3, 3, 4])
np.sum(a==3)                     # 2
np.sum(a==b)                     # 3

a = np.arange(8).reshape(2, 4)   # ndarray([[ 0 1 2 3],
                                 #          [ 4 5 6 7]])

a.sum(axis=0)                    # sum of each column ndarray([4 6 8 10])

a.min(axis=1)                    # min of each row ndarray([0 4])
a.cumsum(axis=1)                 # cumulative sum along each row ndarray([[ 0 1 3 6] [ 4 9 15 22]])

from collections import Counter
a = np.array([1, 2, 2, 3, 1, 2])
common = Counter(a)              # Counter({2: 3, 1: 2, 3: 1})
v = common.most_common(1)[0][0]  # 2

scores = np.array( [ [0.3, 0.7],
                     [0.6, 0.4],
                     [0.4, 0.6]] )
prediction_labels = scores.argmax(axis=1)   # ndarray([1 0 1])
true_labels = np.array([1, 0, 0])

matches = np.sum(prediction_labels == true_labels)
accuracy = np.mean(prediction_labels == true_labels)

prediction_score  = scores[range(scores.shape[0]), prediction_labels]

########## Statistics
a = np.arange(8).reshape(2, 4)   # ndarray([[ 0 1 2 3],
                                 #          [ 4 5 6 7]])

a.mean()                         # 3.5
a.mean(axis=0)                   # ndarray([ 2 3 4 5])

# cov, mean, std, var

########### Functions
a = np.array([1, 5, 4, 2, 8])
index = a.argsort()              # ndarray([0 3 2 1 4])
a[index]                         # ndarray([1 2 4 5 8])

np.exp(B)
np.sqrt(A)
np.add(B, A)

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

x + y
np.add(x, y)

x - y
np.subtract(x, y)

x * y
np.multiply(x, y)

x / y
np.divide(x, y)

np.sqrt(x)

np.prod(x)              # Multiple elements

np.maximum(0, x)        # Replace element smaller than 0 with 0

# all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil,
# clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, inv,
# lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round,
# sort, std, sum, trace, transpose, var, vdot, vectorize, where

########### Concatentate, Stacking
a = np.arange(3)
b = np.arange(3, 6)

np.concatenate((a, b))   # ndarray([0 1 2 3 4 5])

a = np.array([[4, 2], [1, 3]])
v = np.hstack((np.ones((a.shape[0], 1)), a)) # Prepend a 1 on each row

a = np.array([[4, 2], [1, 3]])
b = np.array([[1, 4], [0, 2]])

np.concatenate((a,b), axis=0)  # ndarray([[4 2] [1 3] [1 4] [0 2]])

np.vstack((a, b))        # ndarray([[4 2] [1 3] [1 4] [0 2]])
np.hstack((a, b))        # ndarray([[4 2 1 4] [1 3 0 2]])

np.column_stack((a, b))  # With 2D arrays ndarray([[4 2 1 4] [1 3 0 2]])

np.hstack((np.ones(a.shape[0])[:, np.newaxis], a))  # Prepend a 1 on each row

a = np.array([4, 1])
b = np.array([3, 2])
np.column_stack((a[:, np.newaxis], b[:, np.newaxis]))  # ndarray([[[4 3] [1 2]]])
np.vstack((a[:, np.newaxis], b[:, np.newaxis]))        # ndarray([[[4] [1] [3] [2]]])

np.r_[1:5, 2, 4]                                 # ndarray([1 2 3 4 2 4])

a = np.arange(0, 8, 2)                           # ndarray([0 2 4 6])
b = np.arange(4)                                 # ndarray([0 1 2 3])
np.vstack([a, b])                                # ndarray([[[0 2 4 6] [0 1 2 3]]])
np.hstack([a, b])                                # ndarray([[0 2 4 6 0 1 2 3]])

########### Splitting

a = np.arange(24).reshape((2, 12))
v = np.hsplit(a, 2)                              # Split a into 2 ndarray
v = np.hsplit(a, (3, 4))                         # Split a after the 3rd and 4th column
                                                 # [ndarray([[ 0,  1,  2],
                                                 #        [12, 13, 14]]), array([[ 3],
                                                 #        [15]]), array([[ 4,  5,  6,  7,  8,  9, 10, 11],
                                                 #        [16, 17, 18, 19, 20, 21, 22, 23]])]

########### Copy and view
a = np.arange(12)
b = a                                            # no object created
b is a                                           # True
b.shape = 3, 4
a.shape                                          # (3, 4)


def f(x):  # Mutable object in python is passed by reference
    print(id(x))


c = a.view()
c is a                                           # False
c.base is a                                      # True c is a view of the data owned by a
c.flags.owndata                                  # False
c.shape = 2, 6                                   # Does not impact the shape of a
a.shape                                          # (3, 4)
c[0, 4] = 0                                      # 'a' will also change

s = a[:, 1:4]                                    # Slicing returns a view

d = a.copy()                                     # New object with new data is created.


########## Linear algebra
a = np.array([[1.0, 2.0], [3.0, 4.0]])

np.linalg.inv(a)

u = np.eye(2)
np.trace(u)  # Sum the diganoals

y = np.array([[3.], [8.]])
np.linalg.solve(a, y)

j = np.array([[0.0, -3.0], [2.0, 0.0]])
np.linalg.eig(j)                # Find eign value and vector

######## ix_
# ix_ construct index arrays that will index the cross product.
# a[np.ix_([0,1],[3,2])] returns the array [[a[0,3] a[0,2]], [a[1,3] a[1,2]]].
a = np.arange(10).reshape(2, 5)
                                # array([[0, 1, 2, 3, 4],
                                #       [5, 6, 7, 8, 9]])

grid = np.ix_([0,1], [3, 2])    # (array([[0], [1]]), array([[3, 2]]))

grid[0].shape, grid[1].shape    # (2, 1), (1, 2)
a[grid]                         # [[3 2] [8 7]]

# conversion
# ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat

# Manipulation
# array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack,
# ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack

# Questions
# all, any, nonzero, where

# Ordering
# argmax, argmin, argsort, max, min, ptp, searchsorted, sort

# Operations
# choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum


# Basic Linear Algebra
# cross, dot, outer, linalg.svd, vdot

