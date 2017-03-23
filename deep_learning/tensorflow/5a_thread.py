import tensorflow as tf
import threading
import time
import random

# Thread body: loop until the coordinator request it to stop.
def loop(coord):
  i = 0
  # Check if the coordinate request me to stop.
  while not coord.should_stop():
    i += 1
    print(f"{threading.get_ident()} say {i}")
    time.sleep(random.randrange(4))
    if i == 4:
      # Request the coord to stop all threads.
      coord.request_stop()

# Main thread: create a coordinator.
coord = tf.train.Coordinator()

# Create 4 threads
threads = [threading.Thread(target=loop, args=(coord,)) for i in range(4)]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()

coord.join(threads)