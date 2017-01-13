import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import os


def set_working_dir():
    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    os.chdir(tmp_dir)
    print("Change working directory to", os.getcwd())


set_working_dir()

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


# With regularization
batch_size = 128
num_hidden_nodes1 = 1024
num_hidden_nodes2 = 128

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    lmbda = tf.placeholder(tf.float32)
    global_step = tf.Variable(0)  # count the number of steps taken.

    # Variables.
    weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_hidden_nodes1], stddev=np.sqrt(2.0 / (image_size * image_size))))
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
    weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
    biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
    weights3 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes2, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
    biases3 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    hidden_layer1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, weights2) + biases2)
    logits = tf.matmul(hidden_layer2, weights3) + biases3
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) \
           + lmbda * tf.nn.l2_loss(weights1) + lmbda * tf.nn.l2_loss(weights2) + lmbda * tf.nn.l2_loss(weights3)

    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.5, global_step,
                                               1000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_h1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    valid_h2 = tf.nn.relu(tf.matmul(valid_h1, weights2) + biases2)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_h2, weights3) + biases3)
    test_h1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_h2 = tf.nn.relu(tf.matmul(test_h1, weights2) + biases2)
    test_prediction = tf.nn.softmax(tf.matmul(test_h2, weights3) + biases3)

lmbda_value = 1e-3

num_steps = 9001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("\nInitialized with regularization value: %.5f with decay learning rate" % lmbda_value)
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, lmbda: lmbda_value}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    print("Completed. Trained with regularization value: %.5f with decay learning rate." % lmbda_value)

# Test accuracy:  94.4%
