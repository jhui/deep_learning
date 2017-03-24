import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def model(a, D):
    X = [ item[0] for item in a]
    Y = [ item[1] for item in a]

    test_value = np.arange(0, np.max(Y), 0.2)
    test_data = np.array([ [item**i for i in range(D)] for item in test_value ])
    test_data_holder = tf.placeholder(tf.float64, shape=(test_value.shape[0], D))

    lmbda_holder = tf.placeholder(tf.float64, shape=None)

    x_data = np.array([ [i**j for j in range(D)] for i in X ], dtype='float64')
    y_data = x_data[:, 1]

    print("x shape = {} y shape = {}".format(x_data.shape, y_data.shape))


    W = tf.Variable(tf.random_uniform([D, 1], -0.1, 0.1, dtype=tf.float64))
    y = tf.matmul(x_data, W)
    y = tf.reshape(y, [-1])

    test_y = tf.matmul(test_data_holder, W)

#    lmbda = 1
    lmbda = 1

    # Gradient descent
    loss = tf.reduce_mean(tf.square(tf.subtract(y, y_data))) + lmbda_holder*tf.nn.l2_loss(W)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for step in range(100000):
            sess.run(train, {lmbda_holder:lmbda})
            if step % 1000 == 0:
                print(step, sess.run([loss], {test_data_holder:test_data, lmbda_holder:lmbda}))

        loss, params, out = sess.run([loss, W, test_y], {test_data_holder:test_data, lmbda_holder:lmbda})
        print(params)
        print(f"loss = {loss}")

        plt.colors()
        plt.scatter(X, Y)
        plt.plot(X, X)
        plt.plot(test_value, np.reshape(out, [-1]))
        plt.show()


D = 6

a=[[0, 0], [2, 1.0], [4, 4.8], [6, 6.3], [8, 7.6], [10, 10], [12, 11.6], [14, 12.8], [16, 17.2], [18, 18.4], [20, 20]]
model(a, D)

a=[[0, 0], [1, 1.1], [2, 1.0], [3, 2.5], [4, 4.8], [5, 5.2], [6, 6.3], [7, 7.2], [8, 7.6], [9, 9.1], [10, 10], [11, 11.2], [12, 11.6], [13, 13.3], [14, 12.8], [15, 15.6], [16, 17.2], [17, 17.4], [18, 18.4], [19, 19.4], [20, 20]]
model(a, D)

# 87000 [2.5431744485195127]
# 88000 [2.525734745522529]
# 89000 [223.88938197268865]
# 90000 [195.08231216279583]
# 91000 [3.0582387198108449]
# 92000 [2.4587727305339286]
