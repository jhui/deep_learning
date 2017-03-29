import tensorflow as tf
import numpy as np

### ndarray shape
x = np.array([[2, 3], [4, 5], [6, 7]])
print(x.shape)          # (3, 2)

x = x.reshape((2, 3))
print(x.shape)          # (2, 3)

x = x.reshape((-1))
print(x.shape)          # (6,)

x = x.reshape((6, -1))
print(x.shape)          # (6, 1)

x = x.reshape((-1, 6))
print(x.shape)          # (1, 6)

### Tensor
W = tf.Variable(tf.random_uniform([4, 5], -1.0, 1.0))

print(W.get_shape())    # Get the shape of W (4, 5)
print(tf.shape(W))      # Wrong. tf.shape(W) returns an tensor that in runtime returns the shape.
                        # Tensor("Shape:0", shape=(2,), dtype=int32)

W = tf.reshape(W, [10, 2])
print(W.get_shape())    # (10, 2)

W = tf.reshape(W, [-1])
print(W.get_shape())    # (20,)

W = tf.reshape(W, [5, -1])
print(W.get_shape())    # (5, 4)

shape_op = tf.shape(W)

c = tf.constant([1, 2, 3, 1])
y, _ = tf.unique(c)     # y only contains the unique elements.

print(y.get_shape())    # (?,) This is a dynamic shape. Only know in runtime

y_shape = tf.shape(y)   # Define an op to get the dynamic shape.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(shape_op))  # [5 4]
    print(sess.run(y_shape))   # [3]
