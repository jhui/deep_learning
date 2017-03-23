import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([5.0, 6.0])

# Run the op that 'x' depends on, and then run 'x'
x.initializer.run()

addition = tf.add(x, a)

# addition.eval is shorthand for tf.Session.run (where sess is the current tf.get_default_session.)
print(addition.eval())      # [6. 8.]

# Close the Session when we're done.
sess.close()
