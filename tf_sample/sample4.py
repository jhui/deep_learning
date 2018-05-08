import tensorflow as tf

c = []
a = tf.get_variable(f"a", [2, 2, 3], initializer=tf.random_uniform_initializer(-1, 1))
b = tf.get_variable(f"b", [2, 3, 2], initializer=tf.random_uniform_initializer(-1, 1))

for i, d in enumerate(['/gpu:0', '/gpu:1']):
    with tf.device(d):
        c.append(tf.matmul(a[i], b[i]))

with tf.device('/cpu:0'):
    sum = tf.add_n(c)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(sum))
# [[-0.36499196 -0.07454088]
# [-0.33966339  0.30250686]]