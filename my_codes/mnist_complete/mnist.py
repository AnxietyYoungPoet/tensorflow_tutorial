import tensorflow as tf
import math


def inference(
    input_holder, input_size, hidden1_size, hidden2_size, output_size
):
    # Hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_size, hidden1_size],
                stddev=1.0 / math.sqrt(float(input_size))
            ), name='weights'
        )
        biases = tf.Variable(
            tf.zeros([hidden1_size]), name='biases'
        )
        hidden1 = tf.nn.relu(tf.matmul(input_holder, weights) + biases)

    # Hidden2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden1_size, hidden2_size],
                stddev=1.0 / math.sqrt(float(hidden1_size))
            ), name='weights'
        )
        biases = tf.Variable(
            tf.zeros([hidden2_size]), name='biases'
        )
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Output
    with tf.name_scope('output'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden2_size, output_size],
                stddev=1.0 / math.sqrt(float(input_size))
            ), name='weights'
        )
        biases = tf.Variable(
            tf.zeros([output_size]), name='biases'
        )
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
