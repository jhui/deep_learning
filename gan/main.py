# pip install opencv-python

import os
import sys
import math
from glob import glob
from random import randint

import tensorflow as tf
import numpy as np
import cv2

from gan_utils import *
from gan_env import *


class Color():
    def __init__(self, imgsize=256, batchsize=4):
        self.batch_size = batchsize
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))
        self.image_size = self.output_size = imgsize

        self.gf_dim = 64
        self.df_dim = 64

        self.c_category_dim = 40
        self.c_uniform_dim = 40
        self.c_gaussian_dim = 40

        self.generator_h_dim = 2048
        self.discriminator_h_dim = 1024

        self.input_colors = 1
        self.input_colors2 = 3
        self.output_colors = 3

        self.l1_scaling = 100

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')
        self.d_bn6 = batch_norm(name='d_bn6')
        self.d_bn7 = batch_norm(name='d_bn7')

        # (4, 256, 256, 1)
        self.line_images = tf.placeholder(tf.float32,
                                          [self.batch_size, self.image_size, self.image_size, self.input_colors])

        # (4, 256, 256, 3) color hints
        #self.color_images = tf.placeholder(tf.float32,
        #                                   [self.batch_size, self.image_size, self.image_size, self.input_colors2])

        # (4, 256, 256, 3)
        self.real_images = tf.placeholder(tf.float32,
                                          [self.batch_size, self.image_size, self.image_size, self.output_colors])


        self.c_category = tf.placeholder(tf.float32, [self.batch_size, self.c_category_dim])
        self.c_uniform  = tf.placeholder(tf.float32, [self.batch_size, self.c_uniform_dim])
        self.c_gaussian = tf.placeholder(tf.float32, [self.batch_size, self.c_gaussian_dim])

        # combined_preimage = tf.concat([self.line_images, self.color_images], 3)  # (4, 256, 256, 4)
        combined_preimage = self.line_images

        self.generated_images = self.generator(combined_preimage, self.c_category, self.c_uniform, self.c_gaussian)  # (4, 256, 256, 3)

        self.real_AB = tf.concat([combined_preimage, self.real_images], 3)  # (4, 256, 256, 4)
        self.fake_AB = tf.concat([combined_preimage, self.generated_images], 3)  # (4, 256, 256, 4)

        self.disc_fake, disc_fake_logits, q_category, q_uniform, q_gaussian = self.discriminator(self.fake_AB, generate_q=True)
        tf.get_variable_scope().reuse_variables()

        self.disc_true, disc_true_logits, _ , _ , _ = self.discriminator(self.real_AB)  # (4, 1), (4, 1)
        tf.get_variable_scope()._reuse = False

        cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(q_category + TINY) * self.c_category, reduction_indices=1))
        ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.c_category + TINY) * self.c_category, reduction_indices=1))
        self.q_loss_category = cond_ent + ent

        sd = tf.ones_like(q_uniform)
        epsilon = (self.c_uniform - q_uniform) / (sd + TINY)
        self.q_loss_uniform = tf.reduce_mean(tf.reduce_sum(
            0.5 * np.log(2 * np.pi) + tf.log(sd + TINY) + 0.5 * tf.square(epsilon),
            reduction_indices=1,
        ))

        sd = tf.ones_like(q_gaussian)   # (N, 200)
        epsilon = (self.c_gaussian - q_gaussian) / (sd + TINY)  # (N, 200)
        self.q_loss_gaussian = tf.reduce_mean(tf.reduce_sum(
            0.5 * np.log(2 * np.pi) + tf.log(sd + TINY) + 0.5 * tf.square(epsilon),
            reduction_indices=1,
        ))

        self.q_loss = self.q_loss_category + self.q_loss_uniform + self.q_loss_gaussian

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits))) \
                      + self.l1_scaling * tf.reduce_mean(tf.abs(self.real_images - self.generated_images))
        self.g_loss = self.g_loss_fake

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        self.q_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.q_loss, var_list=self.d_vars + self.g_vars)

    def discriminator(self, image, generate_q = False):
        # image: (N, 256, 256, 7)
        # image: (N, 256, 256, 4)

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))  # (N, 128, 128, 64)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))  # (N, 64, 64, 128)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))  # (N, 32, 32, 256)
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, stride_h=1, stride_w=1, name='d_h3_conv')))  # (N, 32, 32, 512)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')  # (N, 1)

        c_category = c_uniform = c_gaussian = None
        if generate_q:
            q_h0 = lrelu(self.d_bn4(conv2d(h3, self.df_dim * 8, name='d_c1_conv')))  # (N, 16, 16, 512)
            q_h1 = lrelu(self.d_bn5(conv2d(q_h0, self.df_dim * 16, name='d_c2_conv')))  # (N, 8, 8, 1024)
            q_h2 = lrelu(self.d_bn6(conv2d(q_h1, self.df_dim * 16, name='d_c3_conv')))  # (N, 4, 4, 1024)

            q = tf.reshape(q_h2, [self.batch_size, -1])
            q = lrelu(self.d_bn7(linear(q, self.discriminator_h_dim, 'd_q1_lin')))
            logits_category = linear(q, self.c_category_dim, 'd_q2_lin')
            c_category = tf.nn.softmax(logits_category)
            c_uniform  = linear(q, self.c_uniform_dim, 'd_q3_lin')
            c_gaussian = linear(q, self.c_gaussian_dim, 'd_q4_lin')

        return tf.nn.sigmoid(h4), h4, c_category, c_uniform, c_gaussian

    def generator(self, img_in, c_category, c_uniform, c_gaussian):
        # img_in: (N, 256, 256, 4)
        # img_in: (N, 256, 256, 1)
        s = self.output_size  # 256
        s2, s4, s8, s16, s32, s64, s128 = s // 2, s // 4, s // 8, s // 16, s // 32, s // 64, s // 128

        e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv')  # (N, 128, 128, 64)
        e2 = self.bn_g(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'))  # (N, 64, 64, 128)
        e3 = self.bn_g(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'))  # (N, 32, 32, 256)
        e4 = self.bn_g(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'))  # (N, 16, 16, 512)
        e5 = self.bn_g(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'))  # (N, 8, 8, 512)

        e6 = self.bn_g(conv2d(lrelu(e5), self.gf_dim * 16, name='g_e6_conv'))  # (N, 4, 4, 1024)
        e7 = self.bn_g(conv2d(lrelu(e6), self.gf_dim * 32, name='g_e7_conv'))  # (N, 2, 2, 2048)

        l1 = tf.concat([c_category, c_gaussian, c_uniform], 1)
        l2 = self.bn_g(linear(l1, self.generator_h_dim , 'g_l1_lin'))   # (N, 2048)
        l3 = tf.reshape(l2, [self.batch_size, 1, 1, self.generator_h_dim]) # (N, 1, 1, 2048)

        self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(l3), [self.batch_size, s128, s128, self.gf_dim * 16],
                                                 name='g_d1', with_w=True)  # (N, 2, 2, 1024)
        d1 = self.bn_g(self.d1)
        d1 = tf.concat([d1, e7], 3)  # (N, 2, 2, 4096)

        self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1), [self.batch_size, s64, s64, self.gf_dim * 16],
                                                 name='g_d2', with_w=True)  # (N, 4, 4, 1024)
        d2 = self.bn_g(self.d2)
        d2 = tf.concat([d2, e6], 3)  # (N, 4, 4, 2048)

        self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2), [self.batch_size, s32, s32, self.gf_dim * 8],
                                                 name='g_d3', with_w=True)  # (N, 8, 8, 512)
        d3 = self.bn_g(self.d3)
        d3 = tf.concat([d3, e5], 3)  # (N, 8, 8, 1024)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3), [self.batch_size, s16, s16, self.gf_dim * 8],
                                                 name='g_d4', with_w=True)  # (N, 16, 16, 512)
        d4 = self.bn_g(self.d4)
        d4 = tf.concat([d4, e4], 3)  # (N, 16, 16, 1024)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim * 4],
                                                 name='g_d5', with_w=True)  # (N, 32, 32, 256)
        d5 = self.bn_g(self.d5)
        d5 = tf.concat([d5, e3], 3)  # (N, 32, 32, 512)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim * 2],
                                                 name='g_d6', with_w=True)  # (N, 64, 64, 128)
        d6 = self.bn_g(self.d6)
        d6 = tf.concat([d6, e2], 3)  # (N, 64, 64, 256)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim],
                                                 name='g_d7', with_w=True)  # (N, 128, 128, 64)
        d7 = self.bn_g(self.d7)
        d7 = tf.concat([d7, e1], 3)  # (N, 128, 128, 128)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_colors],
                                                 name='g_d8', with_w=True)  # (N, 256, 256, 3)

        return tf.nn.sigmoid(self.d8)

    def imageblur(self, cimg, sampling=False):
        if sampling:
            cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
        else:
            for i in range(30):
                randx = randint(0, 205)
                randy = randint(0, 205)
                cimg[randx:randx + 50, randy:randy + 50] = 255
        return cv2.blur(cimg, (100, 100))

    def train(self):
        self.loadmodel()

        data = glob(os.path.join(IMG_DIR, "*.jpg"))

        base_count = 2
        data[:self.batch_size] = [os.path.join(IMG_DIR, "51.jpg"), os.path.join(IMG_DIR, "1462.jpg"),
                                  os.path.join(IMG_DIR, "13712.jpg"), os.path.join(IMG_DIR, "2868.jpg")]
        data[self.batch_size:self.batch_size*2] = [os.path.join(IMG_DIR, "30018.jpg"), os.path.join(IMG_DIR, "8950.jpg"),
                                                   os.path.join(IMG_DIR, "30324.jpg"), os.path.join(IMG_DIR, "13169.jpg")]

        e_data = [sample_file.replace(DIR_r, DIR_e) for sample_file in data[0:self.batch_size*base_count]]
        if len(e_data) == 0:
            print(f"No JPG image find in {IMG_DIR}")
            exit()

        base = np.array([get_orginal_image(sample_file) for sample_file in data[0:self.batch_size*base_count]])
        base_normalized = base / 255.0

        base_edge = np.array([get_orginal_image(sample_file, color=False) / 255.0 for sample_file in e_data])
        base_edge = np.expand_dims(base_edge, 3)

        # base_colors = np.array([self.imageblur(ba) for ba in base]) / 255.0  # (N, 256, 256, 3)

        base_c_category = sample_category(self.batch_size * base_count, self.c_category_dim)
        base_c_uniform  = sample_uniform(self.batch_size * base_count, self.c_uniform_dim)
        base_c_gaussian = sample_gaussian(self.batch_size * base_count, self.c_gaussian_dim)

        for i in range(base_count):
            ims(os.path.join(RESULT_DIR, f"base{i}.png"),
                merge_color(base_normalized[i*self.batch_size:(i+1)*self.batch_size], [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims(os.path.join(RESULT_DIR, f"base_line{i}.jpg"),
                merge(base_edge[i*self.batch_size:(i+1)*self.batch_size], [self.batch_size_sqrt, self.batch_size_sqrt]))
            # ims(os.path.join(RESULT_DIR, f"base_colors{i}.jpg"),
            #    merge_color(base_colors[i*self.batch_size:(i+1)*self.batch_size], [self.batch_size_sqrt, self.batch_size_sqrt]))

        datalen = len(data)

        for e in range(20000):
            for i in range(datalen // self.batch_size):
                batch_files = data[i * self.batch_size:(i + 1) * self.batch_size]
                e_data = [sample_file.replace(DIR_r, DIR_e) for sample_file in batch_files]

                batch = np.array([get_orginal_image(batch_file) for batch_file in batch_files])
                batch_normalized = batch / 255.0

                batch_edge = np.array([get_orginal_image(sample_file, color=False) / 255.0 for sample_file in e_data])
                batch_edge = np.expand_dims(batch_edge, 3)

                c_category = sample_category(self.batch_size, self.c_category_dim)
                c_uniform  = sample_uniform(self.batch_size, self.c_uniform_dim)
                c_gaussian = sample_gaussian(self.batch_size, self.c_gaussian_dim)

                # batch_colors = np.array([self.imageblur(ba) for ba in batch]) / 255.0

                d_loss, _ = self.sess.run([self.d_loss, self.d_optim],
                                          feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge,
                                                     # self.color_images: batch_colors,
                                                     self.c_category: c_category,
                                                     self.c_gaussian: c_gaussian,
                                                     self.c_uniform: c_uniform})

                g_loss, _, q_loss, _ = self.sess.run([self.g_loss, self.g_optim, self.q_loss, self.q_optim],
                                          feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge,
                                                     # self.color_images: batch_colors,
                                                     self.c_category: c_category,
                                                     self.c_gaussian: c_gaussian,
                                                     self.c_uniform: c_uniform})

                print("%d: [%d / %d] d_loss %f, g_loss %f, q_loss %f" % (e, i, (datalen / self.batch_size), d_loss, g_loss, q_loss))

                if i % 500 == 0:
                    for j in range(base_count):
                        b_start, b_end = j * self.batch_size, (j + 1) * self.batch_size
                        recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: base_normalized[b_start:b_end],
                                                                                     self.line_images: base_edge[b_start:b_end],
                                                                                     # self.color_images: base_colors[b_start:b_end],
                                                                                     self.c_category: base_c_category[b_start:b_end],
                                                                                     self.c_gaussian: base_c_gaussian[b_start:b_end],
                                                                                     self.c_uniform: base_c_uniform[b_start:b_end]})
                        ims(os.path.join(RESULT_DIR, str(e * 100000 + i) + f"_{j}.jpg"),
                            merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))

                if i % 2000 == 0:
                    self.save(e * 100000 + i)

    def sample(self):
        self.loadmodel(False)

        data = glob(os.path.join(SAMPLE_IMG_DIR, "*.jpg"))

        datalen = len(data)

        for i in range(min(100, datalen // self.batch_size)):
            batch_files = data[i * self.batch_size:(i + 1) * self.batch_size]
            batch = np.array([cv2.resize(imread(batch_file), (256, 256)) for batch_file in batch_files])
            batch_normalized = batch / 255.0

            batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255,
                                                         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9,
                                                         C=2) for ba in batch]) / 255.0
            batch_edge = np.expand_dims(batch_edge, 3)

            # batch_colors = np.array([self.imageblur(ba, True) for ba in batch]) / 255.0

            c_category = sample_category(self.batch_size, self.c_category_dim)
            c_uniform = sample_uniform(self.batch_size, self.c_uniform_dim)
            c_gaussian = sample_gaussian(self.batch_size, self.c_gaussian_dim)

            recreation = self.sess.run(self.generated_images,
                                       feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge,
                                                  # self.color_images: batch_colors,
                                                  self.c_category: c_category,
                                                  self.c_gaussian: c_gaussian,
                                                  self.c_uniform: c_uniform})

            ims(os.path.join(RESULT_DIR, "sample_" + str(i) + ".jpg"),
                merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims(os.path.join(RESULT_DIR, "sample_" + str(i) + "_origin.jpg"),
                merge_color(batch_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims(os.path.join(RESULT_DIR, "sample_" + str(i) + "_line.jpg"),
                merge_color(batch_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
            # ims(os.path.join(RESULT_DIR, "sample_" + str(i) + "_color.jpg"),
            #    merge_color(batch_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

    def loadmodel(self, load_discrim=True):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if load_discrim:
            self.saver = tf.train.Saver(max_to_keep=2)
        else:
            self.saver = tf.train.Saver(self.g_vars, max_to_keep=2)

        if self.load():
            print("Checkpoint loaded.")
        else:
            print("No checkpoint loaded.")

    def save(self, step):
        dir = os.path.join(CHECKPOINT_DIR, "model")
        self.saver.save(self.sess, dir, global_step=step)

    def load(self):
        print("Reading checkpoint ...", end=" ")

        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(CHECKPOINT_DIR, ckpt_name))
            return True
        else:
            return False

    def bn_g(self, x):
        return bn(x, prefix="g_bn")

def sample_uniform(batch_size, dim):
    return np.random.uniform(-1.0, 1.0, size=(batch_size, dim))

def sample_gaussian(batch_size, dim):
    return np.random.standard_normal(size=(batch_size, dim))

def sample_category(batch_size, dim):
    return np.random.multinomial(1, dim * [1.0/dim], size=batch_size)

def prepare_dir():
    if not os.path.exists(DATA_ROOT_DIR):
        print(f"Data directory {DATA_ROOT_DIR} not exist")

    if not os.path.exists(APP_ROOT_DIR):
        print(f"App directory {APP_ROOT_DIR} not exist")

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)


if __name__ == '__main__':
    cmd = "train" if len(sys.argv) == 1 else sys.argv[1]
    print("Starting ...")
    prepare_dir()
    if cmd == "train":
        c = Color()
        c.train()
    elif cmd == "sample":
        c = Color(256, 1)
        c.sample()
    else:
        print("Usage: python main.py [train, sample]")


