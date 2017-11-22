# -*- coding: utf-8 -*-
__author__ = 'lijm'
__date__ = '2017/9/13 下午8:14'

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist", one_hot=True)

X_images_input = tf.placeholder(dtype=tf.float32, shape=[None,784], name='X_input')
Y_labels_input = tf.placeholder(dtype=tf.float32, shape=[None,10], name='Y_input')

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
Z = tf.add(tf.matmul(X_images_input, W), b)
Y_output = tf.nn.softmax(Z)
cross_entropy = -tf.reduce_sum(tf.multiply(Y_labels_input, tf.log(Y_output)))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y_labels_input, 1), tf.argmax(Y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.global_variables_initializer()
# init = tf.initialize_all_variables()

data = {
    'iter_num': [],
    'loss': [],
    'predict_train': [],
    'predict_val': [],
}
batch_num = 128  # mini-batch = 128
time_start = time.time() #训练开始
with tf.Session() as sess:
    sess.run(init)
    for i in range(500):
        batch_X, batch_Y = mnist.train.next_batch(batch_num)
        _, loss_batch = sess.run([train, cross_entropy], feed_dict={X_images_input: batch_X, Y_labels_input: batch_Y})

        """
        计算训练数据集的预测精度
        """
        predict_train = sess.run(
            accuracy,
            feed_dict={
                X_images_input:mnist.train.images,
                Y_labels_input:mnist.train.labels
            }
        )
        """
        计算验证数据集的预测精度
        """
        predict_val = sess.run(
            accuracy,
            feed_dict={
                X_images_input: mnist.validation.images,
                Y_labels_input: mnist.validation.labels
            }
        )
        print("for iter_num " + str(i) + " : the predict of train is " + str(predict_train * 100) + "%")
        print("for iter_num " + str(i) + " : the predict of val is " + str(predict_val * 100) + "%")
        data['iter_num'].append(i)
        data['loss'].append(loss_batch)
        data['predict_train'].append(predict_train*100)
        data['predict_val'].append(predict_train*100)

    """
    计算test数据集的预测精度
    """
    predict_test = sess.run(
        accuracy,
        feed_dict={
            X_images_input:mnist.test.images,
            Y_labels_input:mnist.test.labels
        }
    )
    print("for test_dataset : the percent of predict is "+str(predict_test*100)+"%")

time_end = time.time()  #训练结束
print("the time cost : "+str((time_end-time_start)/1000000/60)+"min") # 显示训练时间
df = pd.DataFrame(data) #将字典转化成DataFrame类型
df.plot()  #利用pandas库绘制损失和预测精度曲线
plt.show()
