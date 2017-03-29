import numpy as np
import tensorflow as tf

def model(features, labels, mode):

  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b

  loss = tf.reduce_sum(tf.square(y - labels))
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
x = np.array([1., 2., 3., 4.])
y = np.array([1.5, 3.5, 5.5, 7.5])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)
print(estimator.evaluate(input_fn=input_fn, steps=10))

for name in estimator.get_variable_names():
    print(f'{name} = {estimator.get_variable_value(name)}')

test_data = np.array([5., 6.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": test_data}, None, 2, num_epochs=1)

predictions = estimator.predict(input_fn=input_fn)
for i, p in enumerate(predictions):
    print("Prediction %s: %s" % (i + 1, p))

# {'loss': 6.7292158e-11, 'global_step': 1000}
# W = [ 1.99999637]
# b = [-0.4999892]
# Prediction 1: 11.5000004526
# Prediction 2: 9.50000029689