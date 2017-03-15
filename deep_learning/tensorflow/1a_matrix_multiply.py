import tensorflow as tf

# Construct 2 op nodes (m1, m2) representing 2 matrix.
m1 = tf.constant([[3, 5]])
m2 = tf.constant([[2],[4]])

product = tf.matmul(m1, m2)    # A matrix multiplication op node

with tf.Session() as sess:     # Open a TensorFlow session to execute the graph.
    result = sess.run(product) # Compute the result for “product”
    print(result)              # 3*2+5*4: [[26]]
