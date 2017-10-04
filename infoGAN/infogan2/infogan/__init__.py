import argparse

from os.path import join, realpath, dirname, basename

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from infogan.categorical_grid_plots import CategoricalPlotter
from infogan.tf_utils import (
    scope_variables,
    NOOP,
    load_mnist_dataset,
    run_network,
    leaky_rectify,
)
from infogan.misc_utils import (
    next_unused_name,
    add_boolean_cli_arg,
    create_progress_bar,
    load_image_dataset,
)
from infogan.noise_utils import (
    create_infogan_noise_sample,
    create_gan_noise_sample,
)

SCRIPT_DIR = dirname(realpath(__file__))
PROJECT_DIR = dirname(SCRIPT_DIR)
TINY = 1e-6

def generator_forward(z,
                      network_description,
                      is_training,
                      reuse=None,
                      name="generator",
                      debug=False):
    with tf.variable_scope(name, reuse=reuse):
        # ['fc:1024', 'fc:8x8x256', 'reshape:8:8:256',
        #  'deconv:4:1:256', 'deconv:4:2:256',
        #  'deconv:4:2:128', 'deconv:4:2:64',
        #  'deconv:4:1:1:sigmoid']
        return run_network(z,
                           network_description,
                           is_training=is_training,
                           debug=debug,
                           strip_batchnorm_from_last_layer=True)

def discriminator_forward(img,
                          network_description,
                          is_training,
                          reuse=None,
                          name="discriminator",
                          debug=False):
    with tf.variable_scope(name, reuse=reuse):
        # conv:4:2:64:lrelu, conv:4:2:128:lrelu,
        # conv:4:2:256:lrelu,conv:4:1:256:lrelu,
        # conv:4:1:256:lrelu,
        # fc:1024:lrelu'
        out = run_network(img,
                          network_description,
                          is_training=is_training,
                          debug=debug)
        out = layers.flatten(out)
        prob = layers.fully_connected(
            out,
            num_outputs=1,
            activation_fn=tf.nn.sigmoid,
            scope="prob_projection"
        )

    return {"prob":prob, "hidden":out}


def reconstruct_mutual_info(z_c_categoricals,
                            z_c_continuous,
                            categorical_lambda,
                            continuous_lambda,
                            discriminator_hidden,
                            is_training,
                            reuse=None,
                            name="mutual_info"):
    with tf.variable_scope(name, reuse=reuse):
        out = layers.fully_connected(
            discriminator_hidden,
            num_outputs=128,
            activation_fn=leaky_rectify,
            normalizer_fn=layers.batch_norm,
            normalizer_params={"is_training":is_training}
        )

        num_categorical = sum([true_categorical.get_shape()[1].value for true_categorical in z_c_categoricals])
        num_continuous = z_c_continuous.get_shape()[1].value

        out = layers.fully_connected(
            out,
            num_outputs=num_categorical + num_continuous,
            activation_fn=tf.identity
        )

        # distribution logic
        offset = 0
        loss_q_categorical = None
        for z_c_categorical in z_c_categoricals:
            cardinality = z_c_categorical.get_shape()[1].value
            prob_categorical = tf.nn.softmax(out[:, offset:offset + cardinality])
            loss_q_categorical_new = - tf.reduce_sum(tf.log(prob_categorical + TINY) * z_c_categorical,
                reduction_indices=1
            )
            if loss_q_categorical is None:
                loss_q_categorical = loss_q_categorical_new
            else:
                loss_q_categorical = loss_q_categorical + loss_q_categorical_new
            offset += cardinality

        q_mean = out[:, num_categorical:num_categorical + num_continuous]
        q_sd = tf.ones_like(q_mean)

        epsilon = (z_c_continuous - q_mean) / (q_sd + TINY)
        loss_q_continuous = tf.reduce_sum(
            0.5 * np.log(2 * np.pi) + tf.log(q_sd + TINY) + 0.5 * tf.square(epsilon),
            reduction_indices=1,
        )
        loss_mutual_info = continuous_lambda * loss_q_continuous + categorical_lambda * loss_q_categorical
    return (
        tf.reduce_mean(loss_mutual_info),
        tf.reduce_mean(loss_q_categorical),
        tf.reduce_mean(loss_q_continuous)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--scale_dataset", type=int, nargs=2, default=[28, 28])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--generator_lr", type=float, default=1e-3)
    parser.add_argument("--discriminator_lr", type=float, default=2e-4)
    parser.add_argument("--categorical_lambda", type=float, default=1.0)
    parser.add_argument("--continuous_lambda", type=float, default=1.0)

    parser.add_argument("--categorical_cardinality", nargs="*", type=int, default=[10],
                        help="Cardinality of the categorical variables used in the generator.")
    parser.add_argument("--generator",
                        type=str,
                        default="fc:1024,fc:7x7x128,reshape:7:7:128,deconv:4:2:64,deconv:4:2:1:sigmoid",
                        help="Generator network architecture (call tech support).")
    parser.add_argument("--discriminator",
                        type=str,
                        default="conv:4:2:64:lrelu,conv:4:2:128:lrelu,fc:1024:lrelu",
                        help="Discriminator network architecture, except last layer (call tech support).")
    parser.add_argument("--num_continuous", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--style_size", type=int, default=62)
    parser.add_argument("--plot_every", type=int, default=200,
                        help="How often should plots be made (note: slow + costly).")
    add_boolean_cli_arg(parser, "force_grayscale", default=False, help="Convert images to single grayscale output channel.")
    return parser.parse_args()


def train():
    args = parse_args()

    np.random.seed(args.seed)

    batch_size = args.batch_size         # 64
    n_epochs = args.epochs               # 100
    plot_every = args.plot_every         # 200
    z_noise_size = args.style_size         # 62 Gaussian distributed c
    c_continuous_size = args.num_continuous # 2 Uniform distributed c
    categorical_cardinality = args.categorical_cardinality # [20, 20, 20] 3 categories with value 0 - 19. Represented as 1-hot vector
    generator_desc = args.generator
    discriminator_desc = args.discriminator

    if args.dataset is None:
        assert args.scale_dataset == [28, 28]
        X = load_mnist_dataset()
        dataset_name = "mnist"
    else:
        scaled_image_width, scaled_image_height = args.scale_dataset

        # load pngs and jpegs here
        X = load_image_dataset(
            args.dataset,
            desired_width=scaled_image_width,
            desired_height=scaled_image_height,
            value_range=(0.0, 1.0),
            force_grayscale=args.force_grayscale
        )
        dataset_name = basename(args.dataset.rstrip("/"))

    z_size = z_noise_size + sum(categorical_cardinality) + c_continuous_size  # 124
    sample_noise = create_infogan_noise_sample(
        categorical_cardinality,
        c_continuous_size,
        z_noise_size
    )

    discriminator_lr = tf.get_variable(
        "discriminator_lr", (),
        initializer=tf.constant_initializer(args.discriminator_lr) # discriminator learning rate: 0.0002
    )
    generator_lr = tf.get_variable(
        "generator_lr", (),
        initializer=tf.constant_initializer(args.generator_lr)     # learning rate: 0.001
    )

    n_images, image_height, image_width, n_channels = X.shape # 55,000, 28, 28, 1

    discriminator_lr_placeholder = tf.placeholder(tf.float32, (), name="discriminator_lr")
    generator_lr_placeholder = tf.placeholder(tf.float32, (), name="generator_lr")
    assign_discriminator_lr_op = discriminator_lr.assign(discriminator_lr_placeholder) # Assign placeholder to variable
    assign_generator_lr_op = generator_lr.assign(generator_lr_placeholder)

    ## begin model
    true_images = tf.placeholder(
        tf.float32,
        [None, image_height, image_width, n_channels],
        name="true_images"
    )
    zc_vectors = tf.placeholder(
        tf.float32,
        [None, z_size],
        name="zc_vectors"
    )
    is_training_discriminator = tf.placeholder(
        tf.bool,
        [],
        name="is_training_discriminator"
    )
    is_training_generator = tf.placeholder(
        tf.bool,
        [],
        name="is_training_generator"
    )

    # Generator architecture
    # Fully connected with num_outputs=1024 followed by relu
    # Fully connected with num_outputs=16384 followed by relu
    # Reshape to [8, 8, 256]
    # Deconvolution with nkernels=4, stride=1, num_outputs=256 followed by relu
    # Deconvolution with nkernels=4, stride=2, num_outputs=256 followed by relu
    # Deconvolution with nkernels=4, stride=2, num_outputs=128 followed by relu
    # Deconvolution with nkernels=4, stride=2, num_outputs=64 followed by relu
    # Deconvolution with nkernels=4, stride=1, num_outputs=1 followed by sigmoid
    # Generate (64, 64, 1)
    fake_images = generator_forward(
        zc_vectors,
        generator_desc,
        is_training=is_training_generator,
        name="generator",
        debug=True
    )

    print("Generator produced images of shape %s" % (fake_images.get_shape()[1:]))
    print("")

    # discriminator architecture
    # Convolution with nkernels=4, stride=2, num_outputs=64 followed by lrelu
    # Convolution with nkernels=4, stride=2, num_outputs=128 followed by lrelu
    # Convolution with nkernels=4, stride=2, num_outputs=256 followed by lrelu
    # Convolution with nkernels=4, stride=1, num_outputs=256 followed by lrelu
    # Convolution with nkernels=4, stride=1, num_outputs=256 followed by lrelu
    # Fully connected with num_outputs=1024 followed by lrelu
    # Flatten
    # Fully connected with num_outputs=1 followed by Sigmoid

    discriminator_fake = discriminator_forward(
        fake_images,
        discriminator_desc,
        is_training=is_training_discriminator,
        name="discriminator",
        debug=True
    )
    prob_fake = discriminator_fake["prob"]


    discriminator_true = discriminator_forward(
        true_images,
        discriminator_desc,
        is_training=is_training_discriminator,
        reuse=True,
        name="discriminator",
    )
    prob_true = discriminator_true["prob"]

    # discriminator should maximize:
    loss_discriminator_fake_images = - tf.log(1.0 - prob_fake + TINY)
    loss_discriminator_true_images = - tf.log(prob_true + TINY)
    loss_discriminator = tf.reduce_mean(loss_discriminator_fake_images) + tf.reduce_mean(loss_discriminator_true_images)

    # generator should maximize:
    loss_geneartor = - tf.reduce_mean(tf.log(prob_fake + TINY))

    discriminator_solver = tf.train.AdamOptimizer(
        learning_rate=discriminator_lr,
        beta1=0.5
    )
    generator_solver = tf.train.AdamOptimizer(
        learning_rate=generator_lr,
        beta1=0.5
    )

    discriminator_variables = scope_variables("discriminator")
    generator_variables = scope_variables("generator")

    train_discriminator = discriminator_solver.minimize(loss_discriminator, var_list=discriminator_variables)
    train_generator = generator_solver.minimize(loss_geneartor, var_list=generator_variables)
    discriminator_loss_summary = tf.summary.scalar("loss_discriminator", loss_discriminator)
    generator_loss_summary = tf.summary.scalar("loss_geneartor", loss_geneartor)

    c_categorical_vectors = []
    offset = 0
    for cardinality in categorical_cardinality:
        c_categorical_vectors.append(
            zc_vectors[:, offset:offset + cardinality]
        )
        offset += cardinality

    c_continuous_vector = zc_vectors[:, offset:offset + c_continuous_size]

    loss_mutual_info, loss_q_categorical, loss_q_continuous = reconstruct_mutual_info(
        c_categorical_vectors,
        c_continuous_vector,
        categorical_lambda=args.categorical_lambda,
        continuous_lambda=args.continuous_lambda,
        discriminator_hidden=discriminator_fake["hidden"],
        is_training=is_training_discriminator,
        name="mutual_info"
    )

    mutual_info_variables = scope_variables("mutual_info")
    train_mutual_info = generator_solver.minimize(
        loss_mutual_info,
        var_list=generator_variables + discriminator_variables + mutual_info_variables
    )

    mutual_info_obj_summary = tf.summary.scalar("loss_mutual_info", loss_mutual_info)
    ll_categorical_obj_summary = tf.summary.scalar("ll_categorical_objective", loss_q_categorical)
    ll_continuous_obj_summary = tf.summary.scalar("ll_continuous_objective", loss_q_continuous)
    generator_loss_summary = tf.summary.merge([
        generator_loss_summary,
        mutual_info_obj_summary,
        ll_categorical_obj_summary,
        ll_continuous_obj_summary
    ])


    log_dir = next_unused_name(join(PROJECT_DIR, f"{dataset_name}_log","infogan"))
    journalist = tf.summary.FileWriter(log_dir)
    print(f"Saving tensorboard logs to {log_dir}")

    plotter = CategoricalPlotter(
        categorical_cardinality=categorical_cardinality,
        c_continuous_size=c_continuous_size,
        z_noise_size=z_noise_size,
        journalist=journalist,
        generate=lambda sess, x: sess.run(
            fake_images,
            {zc_vectors: x, is_training_discriminator: False, is_training_generator: False}
        )
    )

    indexes = np.arange(n_images, dtype=np.int32)
    iterations = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            discriminator_epoch_loss = []
            generator_epoch_loss = []
            infogan_epoch_loss = []

            np.random.shuffle(indexes)
            pbar = create_progress_bar(f"epoch {epoch} >> ")

            for index in pbar(range(0, n_images, batch_size)):
                batch = X[indexes[index:index + batch_size]]
                noise = sample_noise(batch_size)
                # Train discriminator
                _, discriminator_summary, discriminator_loss, infogan_loss = sess.run(
                    [train_discriminator, discriminator_loss_summary, loss_discriminator, loss_mutual_info],
                    feed_dict={
                        true_images:batch,
                        zc_vectors:noise,
                        is_training_discriminator:True,
                        is_training_generator:True
                    }
                )

                discriminator_epoch_loss.append(discriminator_loss)
                infogan_epoch_loss.append(infogan_loss)

                # Train generator
                noise = sample_noise(batch_size)
                _, _, generator_summary, generator_loss, infogan_loss = sess.run(
                    [train_generator, train_mutual_info, generator_loss_summary, loss_geneartor, loss_mutual_info],
                    feed_dict={
                        zc_vectors:noise,
                        is_training_discriminator:True,
                        is_training_generator:True
                    }
                )

                journalist.add_summary(discriminator_summary, iterations)
                journalist.add_summary(generator_summary, iterations)
                journalist.flush()

                generator_epoch_loss.append(generator_loss)
                infogan_epoch_loss.append(infogan_loss)

                iterations += 1

                if iterations % plot_every == 0:
                    plotter.generate_images(sess, 10, iteration=iterations)
                    journalist.flush()

            msg = f"epoch %d >> discriminator LL %.2f (lr=%.6f), generator LL %.2f (lr=%.6f) , infogan loss %.2f" % (
                epoch,
                np.mean(discriminator_epoch_loss), sess.run(discriminator_lr),
                np.mean(generator_epoch_loss), sess.run(generator_lr),
                np.mean(infogan_epoch_loss)
            )
            print(msg)