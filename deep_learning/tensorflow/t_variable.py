import tensorflow as tf

### Using variables
# Define variables and its initializer
weights = tf.Variable(tf.random_normal([784, 100], stddev=0.1), name="W")
biases = tf.Variable(tf.zeros([100]), name="b")

counter = tf.Variable(0, name="counter")

# Add an Op to increment a counter
increment = tf.assign(counter , counter + 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  # Execute the init_op to initialize all variables
  sess.run(init_op)

  # Retrieve the value of a variable
  b = sess.run(biases)
  print(b)

### Save and restore variables
counter = tf.Variable(0, name="counter")

increment = tf.assign(counter , counter + 1)

# Saver
saver = tf.train.Saver()
# saver = tf.train.Saver({"my_counter": counter})

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  for _ in range(10):
      sess.run(increment)

  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")

  # Restore
  saver.restore(sess, "/tmp/model.ckpt")

  count = sess.run(counter)
  print(count)

# tf.Variable always create new variable even given the same name.
v1 = tf.Variable(10, name="name1")
v2 = tf.Variable(10, name="name1")
assert(v1 is not v2)
print(v1.name)  # name1:0
print(v2.name)  # name1_1:0

# tf.Variable
def affine(x, shape):
    W = tf.Variable(tf.truncated_normal(shape))
    b = tf.Variable(tf.zeros([shape[1]]))
    model = tf.nn.relu(tf.matmul(x, W) + b)
    return model

x = tf.placeholder(tf.float32, [None, 784])
with tf.variable_scope("n1"):
    n1 = affine(x, [784, 500])
with tf.variable_scope("n1"):
    n2 = affine(x, [784, 500])


### Shareable variables
def affine_reuseable(x, shape):
    W = tf.get_variable("W", shape,
                    initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", [shape[1]],
                    initializer=tf.constant_initializer(0.0))
    model = tf.nn.relu(tf.matmul(x, W) + b)
    return model

nx = tf.placeholder(tf.float32, [None, 784])
with tf.variable_scope("n2"):
    nn1 = affine_reuseable(x, [784, 500])

with tf.variable_scope("n2", reuse=True):
    nn2 = affine_reuseable(x, [784, 500])

### Do not do that
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    # v1 = tf.get_variable("v", [1])
    #  Raises ValueError("... v already exists ...").

with tf.variable_scope("foo", reuse=True):
    # v = tf.get_variable("v", [1])
    #  Raises ValueError("... v does not exists ...").
    pass

### Reuse
with tf.variable_scope("foo"):
    v = tf.get_variable("v2", [1])  # Create a new variable.

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v2")      # reuse/share the variable "foo/v2"
assert v1 == v

with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v3", [1])
    scope.reuse_variables()
    v1 = tf.get_variable("v3")
assert v1 == v

### Scoping
with tf.name_scope("foo1"):
    v1 = tf.get_variable("v", [1])
    v2 = tf.Variable(1, name="v2")

with tf.variable_scope("foo2"):
    v3 = tf.get_variable("v", [1])
    v4 = tf.Variable(1, name="v2")

print(v1.name)  # v:0
print(v2.name)  # foo1/v2:0
print(v3.name)  # foo2/v:0
print(v4.name)  # foo2/v2:0