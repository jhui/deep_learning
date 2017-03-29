import tensorflow as tf

import numpy as np

# Create a linear regressorw with 1 feature "x".
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1., 2., 3., 4.])
y = np.array([1.5, 3.5, 5.5, 7.5])

# Construct an input_fn to pre-process and feed data into the models.
# Create 1000 epochs with batch size = 4.
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)
result = estimator.evaluate(input_fn=input_fn)
print(f"loss = {result['loss']}")

for name in estimator.get_variable_names():
    print(f'{name} = {estimator.get_variable_value(name)}')

# loss = 0.013192394748330116
# linear/x/weight = [[ 1.90707111]]
# linear/bias_weight = [-0.21857721]
