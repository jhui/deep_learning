import tensorflow as tf

global_step = tf.Variable(0)

W = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.0], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

model = W * x + b

loss = tf.reduce_sum(tf.square(model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
gradients, v = zip(*optimizer.compute_gradients(loss))
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [1.5, 3.5, 5.5, 7.5]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(optimizer, {x:x_train, y:y_train})

    l_W, l_b, l_cost  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print(f"W: {l_W} b: {l_b} cost: {l_cost}")
    # W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11

