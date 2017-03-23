import numpy as np
import tensorflow as tf
import time

NUM_THREADS = 2
N_SAMPLES = 5

x = np.random.randn(N_SAMPLES, 4) + 1         # shape (5, 4)
y = np.random.randint(0, 2, size=N_SAMPLES)   # shape (5, )
x2 = np.zeros((N_SAMPLES, 4))

# Define a FIFOQueue which each queue entry has 2 elements of length 4 and 1 respectively
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

# Create an enqueue op to enqueue the [x, y]
enqueue_op1 = queue.enqueue_many([x, y])
enqueue_op2 = queue.enqueue_many([x2, y])

# Create an dequeue op
data_sample, label_sample = queue.dequeue()

# QueueRunner: create a number of threads to enqueue tensors in the queue.
# qr = tf.train.QueueRunner(queue, [enqueue_op1] * NUM_THREADS)
qr = tf.train.QueueRunner(queue, [enqueue_op1, enqueue_op2])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # Launch the queue runner threads.
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    for step in range(20):
        if coord.should_stop():
            break
        one_data, one_label = sess.run([data_sample, label_sample])
        print(f"x = {one_data} y = {one_label}")
    coord.request_stop()
    coord.join(enqueue_threads)

# x = [ 0.90066725 -2.47472358  1.4626869   0.93552333] y = 0
# x = [ 3.27642441  0.59251779  2.4254427   0.99563134] y = 0
# x = [-0.36993721  1.10983336  0.07864232  0.78808331] y = 1
# x = [-1.34663463  0.57584733 -0.45564255 -0.27264795] y = 1
# x = [ 1.41686928  0.31506935  0.8132937   1.0751847 ] y = 0
# x = [ 0.  0.  0.  0.] y = 0
# x = [ 0.  0.  0.  0.] y = 0
# ...