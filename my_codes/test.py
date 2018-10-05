import numpy as np
import tensorflow as tf
import time


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('thread', 16,
                            """How often to run the eval.""")


def gen_CSI():
    Rs = tf.random_normal(
      [1000, 10, 10], stddev=1. / np.sqrt(2.))
    Is = tf.random_normal(
      [1000, 10, 10], stddev=1. / np.sqrt(2.))
    return tf.sqrt(tf.square(Rs) + tf.square(Is))


def gen_weights():
    size = 1000 
    raw_weights = tf.random_normal(
      [int(0.9 * size), 10], stddev=3.3)
    weights_1 = tf.nn.softmax(raw_weights)
    raw_weights_2 = raw_weights[:int(0.08 * size)] * 0.8 / 3.3
    weights_2 = tf.nn.softmax(raw_weights_2)
    weights_3 = tf.ones([int(0.02 * size), 10]) / 10.
    weights = tf.concat([weights_1, weights_2, weights_3], 0)
    return weights


def generate_data():
    H = gen_CSI()
    weights = gen_weights()
    temp1 = tf.reshape(weights, (-1, 10, 1))
    zeros = tf.zeros((1000, 10, 10))
    weights1 = temp1 + zeros 
    weights1_ = weights1[:, :, :, tf.newaxis]
    weights2_ = tf.transpose(weights1_, [0, 2, 1, 3])
    H_ = tf.reshape(H, (-1, 10, 10, 1))
    return [tf.concat([H_, weights1_, weights2_], 3)]


def get_batch_data():
    images = generate_data()
    input_queue = tf.train.slice_input_producer(
      [images], shuffle=False, num_epochs=None)
    image_batch = tf.train.batch(
      input_queue, batch_size=1, num_threads=FLAGS.thread,
      capacity=100, allow_smaller_final_batch=False)
    # print(image_batch)
    return image_batch


def main(argv=None):
    print(FLAGS.thread)
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            images = get_batch_data()
        with tf.train.MonitoredTrainingSession() as sess:
            start = time.time()
            for i in range(1000):
                sess.run(images)
                # print(sess.run(images)[0][0][0][0])
            print(time.time() - start)


if __name__ == '__main__':
    tf.app.run()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())  # 就是这一行
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess, coord)
# try:
#     # while not coord.should_stop():
#     start = time.time()
#     for i in range(1000):
#         ii = sess.run(images)
#         # print(ii.shape)
#     print(time.time() - start)
#         # print(ii)
#         # print(j)
# except tf.errors.OutOfRangeError:
#     print('Done training')
# finally:
#     coord.request_stop()
# coord.join(threads)
# sess.close()
