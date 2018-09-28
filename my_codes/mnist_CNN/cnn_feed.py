# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_CNN
import os


log_dir = 'D:/learning/Python_learning/Python/TensorFlow/my_codes/mnist_CNN/log/'
data_dir = 'D:/learning/Python_learning/Python/TensorFlow/my_codes/MNIST_data'

input_size = 28 * 28
output_size = 10


def run_training():
  data_sets = input_data.read_data_sets(data_dir, one_hot=True)
  with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    y_ = tf.placeholder(tf.float32, shape=(None, output_size))
    dropout_holder = tf.placeholder(tf.float32)
    logits = mnist_CNN.inference(x, output_size, dropout_holder)
    loss = mnist_CNN.loss(logits, y_)
    train_op = mnist_CNN.training(loss, learning_rate=0.001)
    accuracy = mnist_CNN.accuracy(logits, y_)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(20000):
      batch = data_sets.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = sess.run(
          accuracy, feed_dict={
            x: batch[0], y_: batch[1], dropout_holder: 0.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
      sess.run(train_op, feed_dict={x: batch[0], y_: batch[1], dropout_holder: 0.5})
    print("test accuracy %g" % sess.run(
      accuracy, feed_dict={
        x: data_sets.test.images, y_: data_sets.test.labels, dropout_holder: 1.0}))


def main():
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_training()


if __name__ == '__main__':
    main()
