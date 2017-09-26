import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

from info_utils import ProgressBar, plot


class InfoDCGAN(object):
    def __init__(self, config, sess):
        self.input_dim = config.input_dim      # 784
        self.z_dim = config.z_dim              # 14
        self.c_cat = config.c_cat              # 10: Category c - 1 hot vector for 10 label values
        self.c_cont = config.c_cont            # 2: Continuous c
        self.d_update = config.d_update        # 2: Run discriminator twice before generator
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.lr = config.lr                    # Learning rate 0.001
        self.max_grad_norm = config.max_grad_norm  # 40
        self.show_progress = config.show_progress  # False

        self.optimizer = tf.train.AdamOptimizer

        self.checkpoint_dir = config.checkpoint_dir
        self.image_dir = config.image_dir

        home = str(Path.home())
        DATA_ROOT_DIR = os.path.join(home, "dataset", "MNIST_data")
        self.mnist = input_data.read_data_sets(DATA_ROOT_DIR, one_hot=True)

        self.random_seed = 42

        self.X = tf.placeholder(tf.float32, [None, self.input_dim], 'X')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], 'z')
        self.c_i = tf.placeholder(tf.float32, [None, self.c_cat], 'c_cat')
        self.c_j = tf.placeholder(tf.float32, [None, self.c_cont], 'c_cont')
        self.c = tf.concat([self.c_i, self.c_j], axis=1)
        self.z_c = tf.concat([self.z, self.c_i, self.c_j], axis=1)

        self.training = tf.placeholder_with_default(False, shape=(), name='training')

        self.sess = sess

    def z_sampler(self, dim1):
        return np.random.normal(-1, 1, size=[dim1, self.z_dim])

    def c_cat_sampler(self, dim1):
        return np.random.multinomial(1, [0.1] * self.c_cat, size=dim1)

    def c_cont_sampler(self, dim1):
        return np.random.uniform(0, 1, size=[dim1, self.c_cont])

    def leaky_relu(self, z, name=None):
        return tf.maximum(0.2 * z, z, name=name)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def build_generator(self):
        self.G_W1 = tf.Variable(self.xavier_init([self.z_dim + self.c_cat + self.c_cont, 1024]))
        self.G_b1 = tf.Variable(tf.zeros([1024]))
        self.G_W2 = tf.Variable(self.xavier_init([1024, 7 * 7 * 128]))
        self.G_b2 = tf.Variable(tf.zeros([7 * 7 * 128]))
        self.G_W3 = tf.Variable(self.xavier_init([4, 4, 64, 128]))
        self.G_W4 = tf.Variable(self.xavier_init([4, 4, 1, 64]))

        G_layer1 = tf.nn.relu(tf.matmul(self.z_c, self.G_W1) + self.G_b1) # (-1, 26), (26, 1024) -> (-1, 1024)
        G_layer1 = tf.layers.batch_normalization(G_layer1, training=self.training)

        G_layer2 = tf.nn.relu(tf.matmul(G_layer1, self.G_W2) + self.G_b2) # (7 x 7 x 128)
        G_layer2 = tf.layers.batch_normalization(G_layer2, training=self.training)
        G_layer2 = tf.reshape(G_layer2, [-1, 7, 7, 128])                  # (-1, 7, 7, 128)

        # (-1, 7, 7, 128), (4, 4, 64, 128) -> (-1, 14, 14, 64)
        G_layer3 = tf.nn.conv2d_transpose(G_layer2, self.G_W3, [tf.shape(G_layer2)[0], 14, 14, 64], [1, 2, 2, 1],
                                          'SAME')
        G_layer3 = tf.nn.relu(G_layer3)

        # (-1, 14, 14, 64), (4, 4, 1, 64) -> (-1, 28, 28, 1)
        G_layer4 = tf.nn.conv2d_transpose(G_layer3, self.G_W4, [tf.shape(G_layer3)[0], 28, 28, 1], [1, 2, 2, 1], 'SAME')
        G_layer4 = tf.nn.sigmoid(G_layer4)
        G_layer4 = tf.reshape(G_layer4, [-1, 28 * 28])

        # (-1, 28 * 28)
        self.G = G_layer4

    def build_discriminator_and_Q(self):
        self.D_W1 = tf.Variable(self.xavier_init([4, 4, 1, 64]))
        self.D_W2 = tf.Variable(self.xavier_init([4, 4, 64, 128]))
        self.D_W3 = tf.Variable(self.xavier_init([7 * 7 * 128, 1024]))
        self.D_b3 = tf.Variable(tf.zeros([1024]))
        self.D_W4 = tf.Variable(self.xavier_init([1024, 1]))
        self.D_b4 = tf.Variable(tf.zeros([1]))
        self.Q_W4 = tf.Variable(self.xavier_init([1024, 128]))
        self.Q_b4 = tf.Variable(tf.zeros([128]))
        self.Q_W5 = tf.Variable(self.xavier_init([128, self.c_cat + self.c_cont]))
        self.Q_b5 = tf.Variable(tf.zeros([self.c_cat + self.c_cont]))

        # (-1, 784), (4, 4, 1, 64) -> (-1, 14, 14, 64)
        D_real_layer1 = tf.nn.conv2d(tf.reshape(self.X, [-1, 28, 28, 1]), self.D_W1, [1, 2, 2, 1], 'SAME')
        D_real_layer1 = self.leaky_relu(D_real_layer1)

        # (-1, 14, 14, 64), (4, 4, 64, 128) -> (-1, 7, 7, 128) -> (-1, 7*7*128)
        D_real_layer2 = tf.nn.conv2d(D_real_layer1, self.D_W2, [1, 2, 2, 1], 'SAME')
        D_real_layer2 = self.leaky_relu(D_real_layer2)
        D_real_layer2 = tf.layers.batch_normalization(D_real_layer2, training=self.training)
        D_real_layer2 = tf.reshape(D_real_layer2, [-1, 7 * 7 * 128])

        # (-1, 6272), (6271, 1024) -> (-1, 1024)
        D_real_layer3 = tf.matmul(D_real_layer2, self.D_W3) + self.D_b3
        D_real_layer3 = self.leaky_relu(D_real_layer3)
        D_real_layer3 = tf.layers.batch_normalization(D_real_layer3, training=self.training)

        # (-1, 1024), (1024, 1) -> (-1, 1)
        D_real_layer4 = tf.nn.sigmoid(tf.matmul(D_real_layer3, self.D_W4) + self.D_b4)

        D_fake_layer1 = tf.nn.conv2d(tf.reshape(self.G, [-1, 28, 28, 1]), self.D_W1, [1, 2, 2, 1], 'SAME')
        D_fake_layer1 = self.leaky_relu(D_fake_layer1)

        D_fake_layer2 = tf.nn.conv2d(D_fake_layer1, self.D_W2, [1, 2, 2, 1], 'SAME')
        D_fake_layer2 = self.leaky_relu(D_fake_layer2)
        D_fake_layer2 = tf.layers.batch_normalization(D_fake_layer2, training=self.training)
        D_fake_layer2 = tf.reshape(D_fake_layer2, [-1, 7 * 7 * 128])

        D_fake_layer3 = self.leaky_relu(tf.matmul(D_fake_layer2, self.D_W3) + self.D_b3)
        D_fake_layer3 = tf.layers.batch_normalization(D_fake_layer3, training=self.training)

        D_fake_layer4 = tf.nn.sigmoid(tf.matmul(D_fake_layer3, self.D_W4) + self.D_b4)

        # (-1, 1024), (1024, 128) -> (-1, 128)
        Q_layer4 = tf.matmul(D_fake_layer3, self.Q_W4) + self.Q_b4
        Q_layer4 = tf.layers.batch_normalization(Q_layer4, training=self.training)
        Q_layer4 = self.leaky_relu(Q_layer4)

        # (-1, 128), (128, 12) -> (-1, 12)
        Q_layer5 = tf.matmul(Q_layer4, self.Q_W5) + self.Q_b5
        Q_layer5_cat = tf.nn.softmax(Q_layer5[:, :self.c_cat])  # (-1, 10)
        Q_layer5_cont = tf.nn.sigmoid(Q_layer5[:, self.c_cat:]) # (-1, 2)
        Q_c_given_x = tf.concat([Q_layer5_cat, Q_layer5_cont], axis=1) # (-1, 12)

        self.D_real = D_real_layer4
        self.D_fake = D_fake_layer4
        self.Q_c_given_x = Q_c_given_x

    def build_model(self):
        self.build_generator()
        self.build_discriminator_and_Q()

        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake + 1e-8))
        self.D_loss = -tf.reduce_mean(tf.log(self.D_real + 1e-8) + tf.log(1 - self.D_fake + 1e-8))

        cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.Q_c_given_x + 1e-8) * self.c, axis=1))
        ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.c + 1e-8) * self.c, axis=1))
        self.Q_loss = cond_ent + ent

        G_params = [self.G_W1, self.G_W2, self.G_W3, self.G_W4, self.G_b1, self.G_b2]
        D_params = [self.D_W1, self.D_W2, self.D_W3, self.D_W4, self.D_b3, self.D_b4]
        Q_params = [self.Q_W4, self.Q_W5, self.Q_b4, self.Q_b5]

        self.var_list = {'G_W1': self.G_W1, 'G_W2': self.G_W2, 'G_W3': self.G_W3, 'G_W4': self.G_W4, 'G_b1': self.G_b1,
                         'G_b2': self.G_b2, \
                         'D_W1': self.D_W1, 'D_W2': self.D_W2, 'D_W3': self.D_W3, 'D_W4': self.D_W4, 'D_b3': self.D_b3,
                         'D_b4': self.D_b4, \
                         'Q_W4': self.Q_W4, 'Q_W5': self.Q_W5, 'Q_b4': self.Q_b4, 'Q_b5': self.Q_b5}

        G_optimizer = self.optimizer(self.lr)
        G_grads_and_vars = G_optimizer.compute_gradients(self.G_loss, G_params)
        G_clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in G_grads_and_vars]
        self.G_optim = G_optimizer.apply_gradients(G_clipped_grads_and_vars)

        D_optimizer = self.optimizer(self.lr)
        D_grads_and_vars = D_optimizer.compute_gradients(self.D_loss, D_params)
        D_clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in D_grads_and_vars]
        self.D_optim = D_optimizer.apply_gradients(D_clipped_grads_and_vars)

        Q_optimizer = self.optimizer(self.lr)
        Q_grads_and_vars = Q_optimizer.compute_gradients(self.Q_loss, G_params + Q_params)
        Q_clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in Q_grads_and_vars]
        self.Q_optim = Q_optimizer.apply_gradients(Q_clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(var_list=self.var_list)

    def train(self):
        avg_G_loss = 0
        avg_D_loss = 0
        avg_Q_loss = 0
        iterations = int(self.mnist.train.num_examples / self.batch_size)

        if self.show_progress:
            bar = ProgressBar('Train', max=iterations)

        for i in range(iterations):

            if self.show_progress:
                bar.next()

            batch_xs, _ = self.mnist.train.next_batch(self.batch_size)
            feed_dict = {self.X: batch_xs, \
                         self.z: self.z_sampler(self.batch_size), \
                         self.c_i: self.c_cat_sampler(self.batch_size), \
                         self.c_j: self.c_cont_sampler(self.batch_size), \
                         self.training: True}

            for _ in range(self.d_update):
                _, D_loss = self.sess.run([self.D_optim, self.D_loss], feed_dict=feed_dict)
            _, G_loss = self.sess.run([self.G_optim, self.G_loss], feed_dict=feed_dict)
            _, Q_loss = self.sess.run([self.Q_optim, self.Q_loss], feed_dict=feed_dict)

            avg_G_loss += G_loss / iterations
            avg_D_loss += D_loss / iterations
            avg_Q_loss += Q_loss / iterations

        if self.show_progress:
            bar.finish()

        return avg_G_loss, avg_D_loss, avg_Q_loss

    def run(self):

        for epoch in range(self.nepoch):
            avg_G_loss, avg_D_loss, avg_Q_loss = self.train()

            state = {'G Loss': '{:.5f}'.format(avg_G_loss), \
                     'D Loss': '{:.5f}'.format(avg_D_loss), \
                     'Q Loss': '{:.5f}'.format(avg_Q_loss), \
                     'Epoch': epoch}

            print(state)

            if epoch % 5 == 0:
                feed_dict = {self.z: self.z_sampler(16), self.c_i: self.c_cat_sampler(16),
                             self.c_j: self.c_cont_sampler(16)}
                samples = self.sess.run(self.G, feed_dict=feed_dict)
                fig = plot(samples)
                plt.savefig(os.path.join(self.image_dir, '{:04d}.png'.format(epoch)), bbox_inches='tight')
                plt.close(fig)

                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'InfoGAN.model'))

    def generate(self, c_cat, c_cont):
        self.load()

        return self.sess.run(self.G, feed_dict={self.z: self.z_sampler(len(c_cat)), self.c_i: c_cat, self.c_j: c_cont})

    def extract_features(self, X):
        self.load()

        return self.sess.run(self.Q_c_given_x, feed_dict={self.G: X})

    def load(self):
        print('[*] Reading Checkpoints...')
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('[!] No Checkpoint Found')