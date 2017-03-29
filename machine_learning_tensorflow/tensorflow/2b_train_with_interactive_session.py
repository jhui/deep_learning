import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

print("x shape = {} y shape = {}".format(x_data.shape, y_data.shape))

sess = tf.InteractiveSession()
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
init.run()

for step in range(201):
    # Run the op node
    train.run()
    if step % 20 == 0:
        # Evaluate the tensor
        print(step, W.eval(), b.eval())
