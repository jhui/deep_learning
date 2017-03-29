import tensorflow as tf

t1 = [[1, 2], [3, 4]]
t2 = [[5, 6], [7, 8]]
tf.concat([t1, t2], 0) # [[1, 2], [3, 4], [5, 6], [7, 8]]
tf.concat([t1, t2], 1) # [[1, 2, 5, 6], [3, 4, 7, 8]]

value = tf.Variable(tf.zeros([4, 10]))

s1, s2, s3 = tf.split(value, [2, 3, 5], 1)
# s1 shape(4, 2)
# s2 shape(4, 3)
# s3 shape(4, 5)

# Split 'value' into 2 tensors along dimension 1
s0, s1= tf.split(value, num_or_size_splits=2, axis=1)  # s0 shape(4, 5)

# Generate a one hot array using indexes
indexes = tf.Variable(tf.constant([2, 0, -1, 0]))
target = tf.one_hot(indexes, 3, 2, 0)
# [[0 0 2]
# [2 0 0]
# [0 0 0]
# [2 0 0]]

s0 = tf.cast(s0, tf.int32)
s0 = tf.to_int64(s0)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(target))

