import tensorflow as tf

value = tf.Variable(1, name="double")
twice = tf.constant(2)

new_value = tf.multiply(value, twice)
doubler = tf.assign(value, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(value))
    for _ in range(3):
        _, result = sess.run([doubler, value])
        print(result)
