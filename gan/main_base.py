# pip install opencv-python

import os
import sys
import math
from glob import glob
from random import randint
from sys import platform

import tensorflow as tf
import numpy as np
import cv2

from gan_utils import *

if platform == "linux":
    TOP_ROOT = os.path.join(os.sep, "home", "ubuntu", "gan")

    DATA_ROOT_DIR = os.path.join(TOP_ROOT, "dataset")
    APP_ROOT_DIR = os.path.join(TOP_ROOT, "app")
elif platform == "darwin":
    TOP_ROOT = os.path.join(os.sep, "Users", "venice")

    DATA_ROOT_DIR = os.path.join(TOP_ROOT, "dataset", "anime")
    APP_ROOT_DIR = os.path.join(".")
else:
    print("Platform not supported")
    exit()

# DATA_ROOT_DIR : This directory should exist with valid training dataset

CHECKPOINT_DIR = os.path.join(APP_ROOT_DIR, "checkpoint", "tr")
RESULT_DIR = os.path.join(APP_ROOT_DIR, "results")

DIR_o = "imgs"
DIR_e = "imgs_e"
DIR_r = "imgs_r"

ORG_DIR = os.path.join(DATA_ROOT_DIR, DIR_o)  # /Users/venice/dataset/anime/imgs
IMG_DIR = os.path.join(DATA_ROOT_DIR, DIR_r)  # /Users/venice/dataset/anime/imgs_r

class Color():
    def __init__(self, imgsize=256, batchsize=4):
        self.batch_size = batchsize
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))
        self.image_size = self.output_size = imgsize

        self.gf_dim = 64
        self.df_dim = 64

        self.input_colors = 1
        self.input_colors2 = 3
        self.output_colors = 3

        self.l1_scaling = 100

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        # (4, 256, 256, 1)
        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors])

        # (4, 256, 256, 3) color hints
        self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors2])

        # (4, 256, 256, 3)
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_colors])

        combined_preimage = tf.concat([self.line_images, self.color_images], 3) # (4, 256, 256, 4)

        self.generated_images = self.generator(combined_preimage) # (4, 256, 256, 3)

        self.real_AB = tf.concat([combined_preimage, self.real_images], 3)      # (4, 256, 256, 7)
        self.fake_AB = tf.concat([combined_preimage, self.generated_images], 3) # (4, 256, 256, 7)

        self.disc_true, disc_true_logits = self.discriminator(self.real_AB) # (4, 1), (4, 1)
        tf.get_variable_scope().reuse_variables()

        self.disc_fake, disc_fake_logits = self.discriminator(self.fake_AB)
        tf.get_variable_scope()._reuse = False

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits))) \
                        + self.l1_scaling * tf.reduce_mean(tf.abs(self.real_images - self.generated_images))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)


    def discriminator(self, image, y=None):
        # image: (N, 256, 256, 7)

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv')) # (N, 128, 128, 64)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'))) # (N, 64, 64, 128)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'))) # (N, 32, 32, 256)
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, stride_h=1, stride_w=1, name='d_h3_conv'))) # (N, 32, 32, 512)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin') # (N, 1)
        return tf.nn.sigmoid(h4), h4

    def generator(self, img_in):
        # img_in: (N, 256, 256, 4)
        s = self.output_size   # 256
        s2, s4, s8, s16, s32, s64, s128 = s//2, s//4, s//8, s//16, s//32, s//64, s//128

        e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv') # (N, 128, 128, 64)
        e2 = bn(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # (N, 64, 64, 128)
        e3 = bn(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # (N, 32, 32, 256)
        e4 = bn(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')) # (N, 16, 16, 512)
        e5 = bn(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # (N, 8, 8, 512)


        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e5), [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True) # (N, 16, 16, 512)
        d4 = bn(self.d4)
        d4 = tf.concat([d4, e4], 3)  # (N, 16, 16, 1024)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True) # (N, 32, 32, 256)
        d5 = bn(self.d5)
        d5 = tf.concat([d5, e3], 3) # (N, 32, 32, 512)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True) # (N, 64, 64, 128)
        d6 = bn(self.d6)
        d6 = tf.concat([d6, e2], 3) # (N, 64, 64, 256)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True) # (N, 128, 128, 64)
        d7 = bn(self.d7)
        d7 = tf.concat([d7, e1], 3) # (N, 128, 128, 128)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_colors], name='g_d8', with_w=True) # (N, 256, 256, 3)

        return tf.nn.tanh(self.d8)


    def imageblur(self, cimg, sampling=False):
        if sampling:
            cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
        else:
            for i in range(30):
                randx = randint(0,205)
                randy = randint(0,205)
                cimg[randx:randx+50, randy:randy+50] = 255
        return cv2.blur(cimg,(100,100))

    def train(self):
        self.loadmodel()

        data = glob(os.path.join(IMG_DIR, "*.jpg"))
        e_data = [ sample_file.replace(DIR_r, DIR_e) for sample_file in data[0:self.batch_size] ]
        if len(e_data)==0:
            print(f"No JPG image find in {IMG_DIR}")
            exit()
            
        base = np.array([get_orginal_image(sample_file) for sample_file in data[0:self.batch_size]])
        base_normalized = base/255.0

        base_edge = np.array([get_orginal_image(sample_file, color=False)/255.0 for sample_file in e_data])
        base_edge = np.expand_dims(base_edge, 3)

        base_colors = np.array([self.imageblur(ba) for ba in base]) / 255.0   # (N, 256, 256, 3)

        ims(os.path.join(RESULT_DIR, "base.png"),merge_color(base_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims(os.path.join(RESULT_DIR, "base_line.jpg"),merge(base_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims(os.path.join(RESULT_DIR, "base_colors.jpg"),merge_color(base_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

        datalen = len(data)

        for e in range(20000):
            for i in range(datalen // self.batch_size):
                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                e_data = [sample_file.replace(DIR_r, DIR_e) for sample_file in batch_files]

                batch = np.array([get_orginal_image(batch_file) for batch_file in batch_files])
                batch_normalized = batch/255.0

                batch_edge = np.array([get_orginal_image(sample_file, color=False) / 255.0 for sample_file in e_data])
                batch_edge = np.expand_dims(batch_edge, 3)

                batch_colors = np.array([self.imageblur(ba) for ba in batch]) / 255.0

                d_loss, _ = self.sess.run([self.d_loss, self.d_optim], feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge, self.color_images: batch_colors})
                g_loss, _ = self.sess.run([self.g_loss, self.g_optim], feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge, self.color_images: batch_colors})

                print("%d: [%d / %d] d_loss %f, g_loss %f" % (e, i, (datalen/self.batch_size), d_loss, g_loss))

                if i % 500 == 0:
                    recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: base_normalized, self.line_images: base_edge, self.color_images: base_colors})
                    ims(os.path.join(RESULT_DIR, str(e*100000 + i)+".jpg"),merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))

                if i % 2000 == 0:
                    self.save(e*100000 + i)

    def sample(self):
        self.loadmodel(False)

        data = glob(os.path.join(ORG_DIR, "*.jpg"))

        datalen = len(data)

        for i in range(min(100, datalen // self.batch_size)):
            batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
            batch = np.array([cv2.resize(imread(batch_file), (512,512)) for batch_file in batch_files])
            batch_normalized = batch/255.0

            batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in batch]) / 255.0
            batch_edge = np.expand_dims(batch_edge, 3)

            batch_colors = np.array([self.imageblur(ba,True) for ba in batch]) / 255.0

            recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge, self.color_images: batch_colors})
            ims(os.path.join(RESULT_DIR, "sample_"+str(i)+".jpg"), merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims(os.path.join(RESULT_DIR, "sample_"+str(i)+"_origin.jpg"),merge_color(batch_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims(os.path.join(RESULT_DIR, "sample_"+str(i)+"_line.jpg"),merge_color(batch_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims(os.path.join(RESULT_DIR, "sample_"+str(i)+"_color.jpg"),merge_color(batch_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

    def loadmodel(self, load_discrim=True):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if load_discrim:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(self.g_vars)

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
    cmd = "train" if len(sys.argv)==1 else sys.argv[1]

    print("Starting ...")
    prepare_dir()
    if cmd == "train":
        c = Color()
        c.train()
    elif cmd == "sample":
        c = Color(512, 1)
        c.sample()
    else:
        print("Usage: python main.py [train, sample]")


