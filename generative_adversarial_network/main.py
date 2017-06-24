# Some code originated from https://github.com/kvfrans/generative-adversial

import tensorflow as tf
import numpy as np
from gm_ops import *
from gm_utils import *
import os
import time
from glob import glob
from scipy.misc import imsave as ims

DIM = 64
Z_DIM = 100

train = True

learningrate = 0.0002
batchsize = 64
beta1 = 0.5

def discriminator(image):
    d_bn1 = batch_norm(name='d_bn1')
    d_bn2 = batch_norm(name='d_bn2')
    d_bn3 = batch_norm(name='d_bn3')

    h0 = lrelu(conv2d(image, DIM, name='d_h0'))
    h1 = lrelu(d_bn1(conv2d(h0, DIM * 2, name='d_h1')))
    h2 = lrelu(d_bn2(conv2d(h1, DIM * 4, name='d_h2')))
    h3 = lrelu(d_bn3(conv2d(h2, DIM * 8, name='d_h3')))
    h4 = linear(tf.reshape(h3, [batchsize, -1]), 1, scope='d_h4')
    return tf.nn.sigmoid(h4), h4

def generator(z):
    g_bn0 = batch_norm(name='g_bn0')
    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')
    g_bn3 = batch_norm(name='g_bn3')

    z2 = linear(z, DIM * 8 * 4 * 4, scope='g_h0')
    h0 = tf.nn.relu(g_bn0(tf.reshape(z2, [-1, 4, 4, DIM * 8])))
    h1 = tf.nn.relu(g_bn1(conv_transpose(h0, [batchsize, 8, 8, DIM * 4], name="g_h1")))
    h2 = tf.nn.relu(g_bn2(conv_transpose(h1, [batchsize, 16, 16, DIM * 2], name="g_h2")))
    h3 = tf.nn.relu(g_bn3(conv_transpose(h2, [batchsize, 32, 32, DIM * 1], name="g_h3")))
    h4 = conv_transpose(h3, [batchsize, 64, 64, 3], name="g_h4")
    return tf.nn.tanh(h4)

with tf.Session() as sess:
    images = tf.placeholder(tf.float32, [batchsize, DIM, DIM, 3] , name="real_images")
    zin = tf.placeholder(tf.float32, [None, Z_DIM], name="z")

    G = generator(zin)

    with tf.variable_scope("discriminator") as scope:
        D_prob, D_logit = discriminator(images)
        scope.reuse_variables()
        D_fake_prob, D_fake_logit = discriminator(G)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit, labels=tf.ones_like(D_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake_logit)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.ones_like(D_fake_logit)))

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    data = glob("./data/imagenet/tiny-imagenet-100-A/train/**/images/*.JPEG")
    print(len(data))

    d_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=10)

    counter = 1
    start_time = time.time()

    display_z = np.random.uniform(-1, 1, [batchsize, Z_DIM]).astype(np.float32)

    realfiles = data[0:64]
    realim = [get_image(batch_file, 64, 64) for batch_file in realfiles]
    real_img = np.array(realim).astype(np.float32)
    ims("results/imagenet/real.jpg", merge(real_img,[8,8]))

    if train:
        for epoch in range(10):
            batch_idx = (len(data)//batchsize)-2
            for idx in range(batch_idx):
                batch_files = data[idx*batchsize:(idx+1)*batchsize]
                batchim = [get_image(batch_file, 64, 64) for batch_file in batch_files]
                batch_images = np.array(batchim).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [batchsize, Z_DIM]).astype(np.float32)

                sess.run([d_optim],feed_dict={ images: batch_images, zin: batch_z })
                sess.run([g_optim],feed_dict={ zin: batch_z })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, "  % (epoch, idx, batch_idx, time.time() - start_time,))

                if counter % 200 == 0:
                    sdata = sess.run([G],feed_dict={ zin: display_z })
                    print(np.shape(sdata))
                    ims("results/imagenet/"+str(counter)+".jpg", merge(sdata[0],[8,8]))
                    errD_fake = d_loss_fake.eval({zin: display_z})
                    errD_real = d_loss_real.eval({images: batch_images})
                    errG = g_loss.eval({zin: batch_z})
                    print(errD_real + errD_fake)
                    print(errG)
                if counter % 1000 == 0:
                    saver.save(sess, os.getcwd()+"/training/train", global_step=counter)
    else:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
        batch_z = np.random.uniform(-1, 1, [1, Z_DIM]).astype(np.float32)
        batch_z = np.repeat(batch_z, batchsize, axis=0)
        for i in range(Z_DIM):
            edited = np.copy(batch_z)
            edited[:,i] = (np.arange(0.0, batchsize) / (batchsize/2)) - 1
            sdata = sess.run([G],feed_dict={ zin: edited })
            ims("results/imagenet/"+str(i)+".jpg", merge(sdata[0],[8,8]))
