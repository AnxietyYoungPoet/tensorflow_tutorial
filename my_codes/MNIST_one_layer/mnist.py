import tensorflow as tf
import math


def inference(input_holder, output_size):
    logits = tf.layers.dense(
        input_holder, output_size, name='logits',
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    return logits


def loss(logits, labels):
    return tf.losses.softmax_cross_entropy(labels, logits)
    # return -tf.reduce_sum(labels * tf.log(tf.nn.softmax(logits)))


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
