import tensorflow as tf





var = sample_gaussian(2, 10)
with tf.Session() as sess:
    s1 = sess.run([var])
    print(s1)