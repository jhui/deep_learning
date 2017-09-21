import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import math
from random import randint

from gan_utils import *


class Palette():
    def __init__(self, imgsize=256, batchsize=4):
        """
        :param imgsize: default 256x256 image
        :param batchsize: # of images in a batch, default = 4
        """
        print("Loading Palatte")

        self.batch_size = batchsize
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))

        self.image_size = imgsize
        self.output_size = imgsize

        self.z_dim = 64
        self.gf_dim = 64
        self.df_dim = 64

        self.input_colors = 1
        self.output_colors = 3

        bnreset()

        # (4, 256, 256, 1)
        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors])

        # (4, 16, 16, 3)
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size//16, self.image_size//16, self.output_colors])

        with tf.variable_scope("col"):
            z_mean, z_stddev = self.encoder(self.real_images) # (4, 64), (4, 64)
            samples = tf.random_normal([self.batch_size, self.z_dim], 0, 1, dtype=tf.float32) # (4, 64)
            self.guessed_z = z_mean + (z_stddev * samples) # (4, 64)

            # references: line_images,
            self.generated_images = self.generator(self.line_images, self.guessed_z)

        self.g_loss = tf.reduce_mean(tf.abs(self.real_images - self.generated_images)) * 100
        self.l_loss = tf.reduce_mean(
            0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, axis=1))
        self.cost = tf.reduce_mean(self.g_loss + self.l_loss)

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if ('col' in var.name)]
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.cost, var_list=self.g_vars)

    def encoder(self, real_imgs):
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            h0 = lrelu(conv2d(real_imgs, self.df_dim, stride_h=1, stride_w=1, name="e_h0_col"))  # (4, 16, 16, 64)
            h1 = lrelu(bn(conv2d(h0, self.df_dim, stride_h=1, stride_w=1, name="e_h1_col")))  # (4, 16, 16, 64)
            h2 = lrelu(bn(conv2d(h1, self.df_dim, stride_h=1, stride_w=1, name="e_h2_col")))  # 16
            h3 = lrelu(bn(conv2d(h2, self.df_dim, stride_h=1, stride_w=1, name="e_h3_col")))  # 16
            h4 = lrelu(bn(conv2d(h3, self.df_dim, name="e_h4_col")))  # 8
            h5 = lrelu(bn(conv2d(h4, self.df_dim, name="e_h5_col")))  # 4
            mean = linear(tf.reshape(h5, [self.batch_size, -1]), self.z_dim, "e_mean_col")     # (4, 64)
            stddev = linear(tf.reshape(h5, [self.batch_size, -1]), self.z_dim, "e_stddev_col") # (4, 64)
        return mean, stddev

    def generator(self, img_in, z):
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):

            z0 = linear(z, (self.image_size / 64) * (self.image_size / 64) * self.df_dim, "g_z0_col")  # (4, 1024) 4 x 4 x 64
            z1 = tf.reshape(z0, [self.batch_size, self.image_size // 64, self.image_size // 64, self.df_dim]) #(4, 4, 4, 64)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv_col')  # (4, 128, 128, 64)
            e2 = bn(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv_col'))  # (4, 64, 64, 128)
            e3 = bn(conv2d(lrelu(e2), self.gf_dim * 2, name='g_e3_conv_col'))  # (4, 32, 32, 128)
            e4 = bn(conv2d(lrelu(e3), self.gf_dim * 2, name='g_e4_conv_col'))  # (4, 16, 16, 128)
            e5 = bn(conv2d(lrelu(e4), self.gf_dim * 2, name='g_e5_conv_col'))  # (4, 8, 8, 128)
            e6 = bn(conv2d(lrelu(e5), self.gf_dim * 4, name='g_e6_conv_col'))  # (4, 4, 4, 256)
            combined = tf.concat([z1, e6], 3)
            e7 = bn(deconv2d(combined, [self.batch_size, self.image_size // 32, self.image_size // 32, self.gf_dim * 4],
                             name='g_e7_conv_col'))  # (4, 8, 8, 256)
            e8 = deconv2d(lrelu(e7), [self.batch_size, self.image_size // 16, self.image_size // 16, 3],
                          name='g_e8_conv_col')      # (4, 16, 16, 3)

        return tf.nn.tanh(e8)

    def imgprocess(self, cimg, sampling=False):
        num_segs = 16
        seg_len = 256 // num_segs

        seg = np.ones((num_segs, num_segs, 3))
        for x in range(num_segs):
            for y in range(num_segs):
                seg[x:(x + 1), y:(y + 1), 0] = np.average(
                    cimg[x * seg_len:(x + 1) * seg_len, y * seg_len:(y + 1) * seg_len, 0])
                seg[x:(x + 1), y:(y + 1), 1] = np.average(
                    cimg[x * seg_len:(x + 1) * seg_len, y * seg_len:(y + 1) * seg_len, 1])
                seg[x:(x + 1), y:(y + 1), 2] = np.average(
                    cimg[x * seg_len:(x + 1) * seg_len, y * seg_len:(y + 1) * seg_len, 2])
        return seg

    def train(self):
        s = tf.Session()
        s.run(tf.global_variables_initializer())
        self.loadmodel(s)

        data = glob(os.path.join("/Users/venice/imgs", "*.jpg"))
        print(data[0])
        base = np.array([get_image(sample_file) for sample_file in data[0:self.batch_size]]) # (4, 256, 256, 3)

        base_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255,
                                                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for
                              ba in base]) / 255.0    # (4, 256, 256)
        base_edge = np.expand_dims(base_edge, 3)      # (4, 256, 256, 1)

        base_colors = np.array([self.imgprocess(ba) for ba in base]) / 255.0  # (4, 16, 16, 3)

        ims("results/base_line.jpg", merge(base_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims("results/base_colors.jpg",
            merge_color(np.array([cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST) for x in base_colors]),
                        [self.batch_size_sqrt, self.batch_size_sqrt]))

        datalen = len(data)

        for e in range(20000):
            for i in range(datalen // self.batch_size):
                batch_files = data[i * self.batch_size:(i + 1) * self.batch_size]
                batch = np.array([get_image(batch_file) for batch_file in batch_files]) # (4, 256, 256, 3)

                batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255,
                                                             cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9,
                                                             C=2) for ba in batch]) / 255.0
                batch_edge = np.expand_dims(batch_edge, 3)  # (4, 256, 256, 1)

                batch_colors = np.array([self.imgprocess(ba) for ba in batch]) / 255.0 # (4, 16, 16, 3)

                g_loss, l_loss, _ = self.sess.run([self.g_loss, self.l_loss, self.g_optim],
                                                  feed_dict={self.real_images: batch_colors,
                                                             self.line_images: batch_edge})

                print("%d: [%d / %d] l_loss %f, g_loss %f" % (e, i, (datalen / self.batch_size), l_loss, g_loss))

                if i % 100 == 0:
                    recreation = self.sess.run(self.generated_images,
                                               feed_dict={self.real_images: base_colors, self.line_images: base_edge}) # (4, 16, 16, 3)
                    print(recreation.shape)
                    ims("results/" + str(e * 100000 + i) + "_base.jpg", merge_color(
                        np.array([cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST) for x in recreation]),
                        [self.batch_size_sqrt, self.batch_size_sqrt]))

                    recreation = self.sess.run(self.generated_images,
                                               feed_dict={self.real_images: batch_colors, self.line_images: batch_edge})
                    ims("results/" + str(e * 100000 + i) + ".jpg", merge_color(
                        np.array([cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST) for x in recreation]),
                        [self.batch_size_sqrt, self.batch_size_sqrt]))
                    ims("results/" + str(e * 100000 + i) + "_line.jpg",
                        merge(batch_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
                    ims("results/" + str(e * 100000 + i) + "_original.jpg", merge_color(
                        np.array([cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST) for x in batch_colors]),
                        [self.batch_size_sqrt, self.batch_size_sqrt]))

                if i % 1000 == 0:
                    self.save("./checkpoint", e * 100000 + i)

    def loadmodel(self, sess, load_discrim=True):
        self.sess = sess

        if load_discrim:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(self.g_vars)
            print([v.name for v in self.g_vars])

        if self.load("./checkpoint"):
            print("Loaded")
        else:
            print("Load failed")

    def sample(self):
        s = tf.Session()
        s.run(tf.initialize_all_variables())
        self.loadmodel(s, False)

        data = glob(os.path.join("imgs", "*.jpg"))

        datalen = len(data)

        for i in range(min(100, datalen // self.batch_size)):
            batch_files = data[i * self.batch_size:(i + 1) * self.batch_size]
            batch = np.array([cv2.resize(imread(batch_file), (256, 256)) for batch_file in batch_files])
            batch_normalized = batch / 255.0

            random_z = np.random.normal(0, 1, [self.batch_size, self.z_dim])

            batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255,
                                                         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9,
                                                         C=2) for ba in batch]) / 255.0
            batch_edge = np.expand_dims(batch_edge, 3)

            recreation = self.sess.run(self.generated_images,
                                       feed_dict={self.line_images: batch_edge, self.guessed_z: random_z})
            ims("results/sample_" + str(i) + ".jpg",
                merge_color(np.array([cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST) for x in recreation]),
                            [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("results/sample_" + str(i) + "_origin.jpg",
                merge_color(batch_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("results/sample_" + str(i) + "_line.jpg",
                merge_color(batch_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))

    def save(self, checkpoint_dir, step):
        model_name = "model"
        model_dir = "tr_colors"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        """
        :param checkpoint_dir: checkpoint_dir/tr_colors stored the checkpoint file.
        :return: True if checkpoint restored. False otherwise.
        """
        print(" [*] Reading checkpoint...")

        model_dir = "tr_colors"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py [train, sample]")
    else:
        cmd = sys.argv[1]
        if cmd == "train":
            c = Palette()
            c.train()
        elif cmd == "sample":
            c = Palette(256, 1)
            c.sample()
        else:
            print("Usage: python main.py [train, sample]")
