# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist
import time
import os


log_dir = 'D:/learning/Python_learning/Python/\
TensorFlow/my_codes/mnist_complete/log/'

data_dir = 'D:/learning/Python_learning/Python/\
TensorFlow/my_codes/mnist_complete/MNIST_data/'
input_size = 28 * 28
label_classes = 10


def generate_placeholder(batch_size, input_size):
    input_holder = tf.placeholder(
        tf.float32, shape=(batch_size, input_size)
    )
    label_holder = tf.placeholder(
        tf.int32, shape=(batch_size)
    )
    return input_holder, label_holder


def fill_feed_dict(
    data_set, input_holder, label_holder, batch_size
):
    input_feed, label_feed = data_set.next_batch(batch_size)
    feed_dict = {
        input_holder: input_feed,
        label_holder: label_feed,
    }
    return feed_dict


def do_eval(
    sess, eval_correct, input_holder,
    label_holder, data_set, batch_size
):
    true_count = 0
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(
            data_set, input_holder, label_holder, batch_size
        )
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print(
        'Num examples: %d Num correct: %d Precision @ 1: %0.04f' %
        (num_examples, true_count, precision)
    )


def run_training():
    data_sets = input_data.read_data_sets(data_dir)
    with tf.Graph().as_default():
        input_holder, label_holder = generate_placeholder(50, input_size)
        logits = mnist.inference(
            input_holder, input_size, 128, 32, label_classes
        )
        loss = mnist.loss(logits, label_holder)
        train_op = mnist.training(loss, 0.01)
        eval_correct = mnist.evaluation(logits, label_holder)
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        sess.run(init)
        for step in range(2000):
            start_time = time.time()
            feed_dict = fill_feed_dict(
                data_sets.train, input_holder, label_holder, 50
            )
            _, loss_value = sess.run(
                [train_op, loss], feed_dict=feed_dict
            )
            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (
                    step, loss_value, duration
                ))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == 2000:
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('Training Data Eval:')
                do_eval(
                    sess, eval_correct, input_holder,
                    label_holder, data_sets.train, 50
                )
                print('Validation Data Eval:')
                do_eval(
                    sess, eval_correct, input_holder,
                    label_holder, data_sets.validation, 50
                )
                print('Test Data Eval:')
                do_eval(
                    sess, eval_correct, input_holder,
                    label_holder, data_sets.test, 50
                )


def main():
    run_training()


if __name__ == '__main__':
    main()
