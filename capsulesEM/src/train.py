"""A train script for matrix capsule with EM routing."""

from em_datasets import mnist
import capsules

from settings import *


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    NUM_STEPS_PER_EPOCH = mnist.NUM_TRAIN_EXAMPLES // FLAGS.batch_size # 60,000/24 = 2500

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()

        # images shape (24, 28, 28, 1), labels shape (24, 10)
        images, labels = mnist.inputs(data_directory=FLAGS.data_dir, is_training=True, batch_size=FLAGS.batch_size)

        poses, activations = capsules.nets.capsules_net(images, num_classes=10, iterations=3, name='capsulesEM-V0')


        # activations = tf.Print(activations, [activations.shape, activations[0, ...]], 'activations', summarize=20)

        # margin schedule
        # margin increase from 0.2 to 0.9 after margin_schedule_epoch_achieve_max
        margin_schedule_epoch_achieve_max = 10.0
        margin = tf.train.piecewise_constant(
            tf.cast(global_step, dtype=tf.int32),
            boundaries=[
                int(NUM_STEPS_PER_EPOCH * margin_schedule_epoch_achieve_max * x / 7) for x in range(1, 8)
            ],
            values=[
                x / 10.0 for x in range(2, 10)
            ]
        )

        loss = capsules.nets.spread_loss(
            labels, activations, margin=margin, name='spread_loss'
        )

        # tf.summary.scalar(
        #   'losses/cross_entropy_loss', loss
        # )
        tf.summary.scalar(
            'losses/spread_loss', loss
        )

        # TODO: set up a learning_rate decay
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001
        )

        train_tensor = slim.learning.create_train_op(
            loss, optimizer, global_step=global_step, clip_gradient_norm=4.0
        )

        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            log_every_n_steps=10,
            save_summaries_secs=60,
            saver=tf.train.Saver(max_to_keep=100),
            save_interval_secs=600,
            # yg: add session_config to limit gpu usage and allow growth
            session_config=tf.ConfigProto(
                # device_count = {
                #   'GPU': 0
                # },
                gpu_options={
                    'allow_growth': 0,
                    # 'per_process_gpu_memory_fraction': 0.01
                    'visible_device_list': '0'
                },
                allow_soft_placement=True,
                log_device_placement=False
            )
        )


if __name__ == "__main__":
    tf.app.run()
