import tensorflow as tf

value = tf.Variable(1, name="double")
twice = tf.constant(2)

new_value = tf.multiply(value, twice)
doubler = tf.assign(value, new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(value))
    for _ in range(3):
        sess.run(doubler)
        print(sess.run(value))
