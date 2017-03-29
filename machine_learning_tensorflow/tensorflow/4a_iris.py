"""Example of DNNClassifier for Iris plant dataset."""

from sklearn import metrics
from sklearn import model_selection

import tensorflow as tf

def main(unused_argv):
  ### Loading dataset
  # iris dataset contains 150 samples.
  iris = tf.contrib.learn.datasets.load_dataset('iris')
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)
  print(x_train.shape)    # (120, 4) - 120 samples
  print(x_test.shape)     # (30, 4)

  ### Define the model feature and the model
  # Define the features used in the DNN model: 4 features with real value
  feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

  ### Train and predict
  # Fit
  classifier.fit(x_train, y_train, steps=200)

  # Predict
  predictions = list(classifier.predict(x_test, as_iterable=True)) # a list of 30 elements. 30 predictions for 30 samples.
  score = metrics.accuracy_score(y_test, predictions)
  print(f"Accuracy: {score:f}")


if __name__ == '__main__':
  tf.app.run()

