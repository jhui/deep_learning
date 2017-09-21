import tensorflow as tf
import numpy as np
import cv2


class batch_norm():
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None,
                                            epsilon=self.epsilon, scale=True, scope=self.name)


batchnorm_count = 0


def bnreset():
    global batchnorm_count
    batchnorm_count = 0


def bn(x):
    global batchnorm_count
    batch_object = batch_norm(name=("bn" + str(batchnorm_count)))
    batchnorm_count += 1
    return batch_object(x)


def conv2d(input, output_dim, filter_h=5, filter_w=5, stride_h=2, stride_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_h, filter_w, input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input, output_shape, filter_h=5, filter_w=5, stride_h=2, stride_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [filter_h, filter_w, output_shape[-1], input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=[1, stride_h, stride_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input, output_size, scope="Linear", stddev=0.02, bias_start=0.0, with_w=False):
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input, matrix) + bias


def get_orginal_image(image_path, color=True):
    return np.array(imread(image_path, color))

def get_image(image_path):
    return transform(imread(image_path))


def transform(image, npx=512, is_crop=True):
    cropped_image = cv2.resize(image, (256, 256))

    return np.array(cropped_image)


def imread(path, color=True):
    flag = cv2.IMREAD_COLOR
    if color==False:
        flag = cv2.IMREAD_GRAYSCALE

    readimage = cv2.imread(path, flag)
    return readimage


def merge_color(images, size):
    """
        Merge size[0] x size[1] RGB images into one single image to display result.
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i, j = divmod(idx, size[1])
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def merge(images, size):
    """
        Merge size[0] x size[1] images into one single image to display result.
    """
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i, j = divmod(idx, size[1])
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i, j = divmod(idx, size[1])
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


def ims(name, img):
    print("Saving " + name)
    cv2.imwrite(name, img * 255)
