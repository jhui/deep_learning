import tensorflow as tf

# Create a queue just for the filenames which leads to running multiple threads of reader.
filename_queue = tf.train.string_input_producer(["iris_training.csv", "iris_training2.csv", "iris_training3.csv"])

# Define the reader with input from the filename queue.
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the decoded result.
record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0]]
# Decode each line into CSV data.
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)

features = tf.stack([col1, col2, col3, col4])
labels = tf.stack([col5])

with tf.Session() as sess:
 coord = tf.train.Coordinator()
 # Start all QueueRunners added into the graph.
 threads = tf.train.start_queue_runners(coord=coord)
 for _ in range(200):
     # Read one line of data at a time
     # d_features, d_label = sess.run([features, col5])
     # print(f"{d_features} {d_label}")

     min_after_dequeue = 10
     batch_size = 2
     capacity = min_after_dequeue + 3 * batch_size
     # Use shuffle_batch_join for more than 1 reader
     # Use shuffle_batch for 1 reader but possibly more than 1 thread.
     example_batch, label_batch = tf.train.shuffle_batch_join(
          [[features, labels]], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
     # example_batch, label_batch = tf.train.shuffle_batch(
     #     [features, labels], batch_size=batch_size, capacity=capacity,
     #     min_after_dequeue=min_after_dequeue)
     # example_batch : shape(2, 4)
     # label_batch : shape(2, 1)

 coord.request_stop()
 coord.join(threads)

