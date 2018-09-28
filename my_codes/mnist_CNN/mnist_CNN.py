import tensorflow as tf


def inference(x, output_size, dropout_rate):
  conv1 = tf.layers.conv2d(
    tf.reshape(x, [-1, 28, 28, 1]), 32, 5, 1, padding='same', name='conv1', activation=tf.nn.relu,
    kernel_initializer=tf.contrib.layers.xavier_initializer())
  pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same', name='pool1')
  conv2 = tf.layers.conv2d(
    pool1, 64, 5, 1, padding='same', name='conv2', activation=tf.nn.relu,
    kernel_initializer=tf.contrib.layers.xavier_initializer())
  pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same', name='pool2')
  pool_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name='reshape')
  fc = tf.layers.dense(
    pool_flat, 1024, name='fc',
    kernel_initializer=tf.contrib.layers.xavier_initializer())
  drop_out = tf.layers.dropout(fc, rate=dropout_rate, name='dropout')
  logits = tf.layers.dense(
    drop_out, output_size, name='logits',
    kernel_initializer=tf.contrib.layers.xavier_initializer())
  return logits


def loss(logits, labels):
  return tf.losses.softmax_cross_entropy(labels, logits)


def training(loss, learning_rate):
  optimizer = tf.train.AdamOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def accuracy(logits, labels):
  correct = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))
