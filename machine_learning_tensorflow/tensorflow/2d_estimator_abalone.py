#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""DNNRegressor with custom estimator for abalone dataset."""
# Jonathan Hui. This file has been modified from the original Tensorflow tutorial example

import argparse
import sys
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import urllib.request

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_RATE = 0.001

def maybe_download_file(filename, url):
    if not filename:
        download_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve(
            url,
            download_file.name)
        filename = download_file.name
        download_file.close()
        print(f"{url} is downloaded to {filename}")
    return filename

def maybe_download(train_data, test_data, predict_data):
  """Maybe downloads training data and returns train and test file names."""
  train_file_name = maybe_download_file(train_data, "http://download.tensorflow.org/data/abalone_train.csv")
  test_file_name = maybe_download_file(test_data, "http://download.tensorflow.org/data/abalone_test.csv")
  predict_file_name = maybe_download_file(predict_data, "http://download.tensorflow.org/data/abalone_predict.csv")

  return train_file_name, test_file_name, predict_file_name


def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

  first_hidden_layer = tf.contrib.layers.relu(features, 10)
  second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)
  output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}

  loss = tf.losses.mean_squared_error(targets, predictions)

  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(targets, tf.float64), predictions)
  }

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

  return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load datasets
  abalone_train, abalone_test, abalone_predict = maybe_download(
      FLAGS.train_data, FLAGS.test_data, FLAGS.predict_data)

  # Training examples
  training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)

  # Test examples
  test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

  # Set of 7 examples for which to predict abalone ages
  prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

  # Set model params
  model_params = {"learning_rate": LEARNING_RATE}

  # Instantiate Estimator
  nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)



  # Fit
  nn.fit(x=training_set.data, y=training_set.target, steps=5000)

  # Score accuracy
  ev = nn.evaluate(x=test_set.data, y=test_set.target, steps=1)
  print("Loss: %s" % ev["loss"])
  print("Root Mean Squared Error: %s" % ev["rmse"])

  # Print out predictions
  predictions = nn.predict(x=prediction_set.data, as_iterable=True)
  for i, p in enumerate(predictions):
    print("Prediction %s: %s" % (i + 1, p["ages"]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_data", type=str, default="", help="Path to the training data.")
  parser.add_argument(
      "--test_data", type=str, default="", help="Path to the test data.")
  parser.add_argument(
      "--predict_data",
      type=str,
      default="",
      help="Path to the prediction data.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)