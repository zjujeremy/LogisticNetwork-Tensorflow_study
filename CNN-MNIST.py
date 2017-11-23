# -*- coding:utf-8 -*-
__author__ = 'lijm'
__date__ = '2017/11/22 下午4:18'

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

def weight_init(shape):
    weight_init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(weight_init, dtype=tf.float32)

def bais_init(shape):
    bais_init = tf.constant(0.1, shape=shape)
    return tf.Variable(bais_init, dtype=tf.float32)

def conv2d(input, filter):
    return tf.nn.conv2d(input, filter,
                        padding='SAME',
                        strides=[1, 1, 1, 1])

def maxpooling_2X2(input):
    return tf.nn.max_pool(value=input,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./mnist", one_hot=True)

    sess = tf.InteractiveSession()

    x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='x_input')
    y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y_labels')
    x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

    """
        The first layer of convolution and max-pooling
    """
    conv1_w = weight_init([5, 5, 1, 32])
    conv1_b = bais_init([32])

    conv1_res = conv2d(x_image, conv1_w)
    conv1_relu = tf.nn.relu(tf.add(conv1_res, conv1_b))
    conv1_pool = maxpooling_2X2(conv1_relu)

    """
        The second layer of convolution and max-pooling
    """
    conv2_w = weight_init([5, 5, 32, 64])
    conv2_b = bais_init([64])

    conv2_res = conv2d(conv1_pool, conv2_w)
    conv2_relu = tf.nn.relu(tf.add(conv2_res, conv2_b))
    conv2_pool = maxpooling_2X2(conv2_relu)

    """
        the first layer of full connection
    """
    fc1_w = weight_init([7*7*64, 1024])
    fc1_b = bais_init([1024])

    conv2_pool_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
    fc1_relu = tf.nn.relu(tf.matmul(conv2_pool_flat, fc1_w)+fc1_b)

    """
        the output layer of softmax
    """
    softmax_w = weight_init([1024, 10])
    softmax_b = bais_init([10])
    y_output = tf.nn.softmax(tf.matmul(fc1_relu, softmax_w)+softmax_b)

    cross_entropy = -tf.reduce_sum(y*tf.log(y_output))
    train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    accurate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_output, axis=1), tf.argmax(y, axis=1)), dtype=tf.float32))

    init = tf.global_variables_initializer()
    # init = tf.initialize_all_variables()

    start_time = time.time()

    train_process = {
        # 'batch_loss':[],
        'accurate_train_set':[],
        'accurate_validation':[]
    }
    sess.run(init)
    for i in range(300):
        print("=========Iter_num"+str(i)+"==========")
        batch_x, batch_y = mnist.train.next_batch(128)
        _, batch_cost = sess.run([train, cross_entropy], feed_dict={x:batch_x, y:batch_y})

        """
            对训练集的预测精确度
        """
        accurate_train = sess.run(
            accurate,
            feed_dict={
                x:mnist.train.images[:100],
                y:mnist.train.labels[:100]
            }
        )
        print("the accurate of train :"+str(accurate_train*100)+"%")
        """
            对验证集的预测精确度
        """
        accurate_val = sess.run(
            accurate,
            feed_dict={
                x:mnist.validation.images[:100],
                y:mnist.validation.labels[:100]
            }
        )
        print("the accurate of validation :" + str(accurate_val * 100) + "%")
        # train_process['batch_loss'].append(batch_cost)
        train_process['accurate_train_set'].append(accurate_train)
        train_process['accurate_validation'].append(accurate_val)

    df = pd.DataFrame(train_process)
    df.plot()
    plt.show()
    end_time = time.time()
    """
        对test集的预测精确度
    """
    accurate_test = sess.run(
        accurate,
        feed_dict={
            x:mnist.test.images[:2000],
            y:mnist.test.labels[:2000]
        }
    )
    print("the accurate of test data set :" + str(accurate_test * 100) + "%")




