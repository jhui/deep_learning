import os

import pprint
import tensorflow as tf

from InfoDCGAN import InfoDCGAN

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer('input_dim', 784, 'dimension of the discriminator input placeholder [784]')
flags.DEFINE_integer('z_dim', 14, 'dimension of the generator input noise variable z [14]')
flags.DEFINE_integer('c_cat', 10, 'dimension of the categorical latent code [10]')
flags.DEFINE_integer('c_cont', 2, 'dimension of the continuous latent code [2]')
flags.DEFINE_integer('d_update', 2,
                     'update the discriminator weights [d_update] times per generator/Q network update [2]')
flags.DEFINE_integer('batch_size', 128, 'batch size to use during training [128]')
flags.DEFINE_integer('nepoch', 100, 'number of epochs to use during training [100]')
flags.DEFINE_float('lr', 0.001, 'learning rate of the optimizer to use during training [0.001]')
flags.DEFINE_float('max_grad_norm', 40, 'clip L2-norm of gradients to this threshold [40]')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'checkpoint directory [./checkpoints]')
flags.DEFINE_string('image_dir', './images', 'directory to save generated images to [./images]')
flags.DEFINE_bool('show_progress', False, 'print progress [False]')

FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.image_dir):
        os.makedirs(FLAGS.image_dir)

    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:

        model = InfoDCGAN(FLAGS, sess)
        model.build_model()
        model.run()


if __name__ == '__main__':
    tf.app.run()