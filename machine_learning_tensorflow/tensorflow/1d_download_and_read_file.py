import tempfile

import tensorflow as tf
import urllib.request
import numpy as np

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

def maybe_download(train_data):
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_train.csv",
        train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)
  return train_file_name


training_local_file = ""
training_local_file = maybe_download(training_local_file)

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
  filename=training_local_file, target_dtype=np.int, features_dtype=np.float64)

print(f"data shape = {training_set.data.shape}")      # (3320, 7)
print(f"label shape = {training_set.target.shape}")   # (3320,)

pass