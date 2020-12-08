
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
import codecs
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import MultipleLocator

def get_datacsv(x_file, y_file): #预处理原始数据集，生成x,y两个数据集，其中标签y采用one-hot编码
    # 数据集若已经存在，直接读取
    if os.path.exists(x_file) and os.path.exists(y_file):
        print("Data has been generated.")
        x = pd.read_csv(x_file)
        y = pd.read_csv(y_file)
        return x, y

    data=pd.read_csv('data/label_vec_16.csv')
    x = data.drop('label', axis=1)  # 去掉label列
    labels = data.label
    y = np.zeros([len(labels), 2])
    for i in range(x.shape[0]):  # 标签y采用one-hot编码
        y[i][int(labels[i]) - 1] = 1
    y = pd.DataFrame(columns=['y1', 'y2'], data=y)
    x.to_csv(x_file, index=False)
    y.to_csv(y_file, index=False)  # 预处理数据保存，下次使用可直接读取，不必再进行计算
    return x, y


learning_rate = 0.0001
keep_prob_rate = 0.7
max_epoch = 100
batch_size = 10

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    outputs=tf.nn.conv2d(input=x,filters=W,strides=[1,1,1,1],padding='SAME')
    return outputs

def max_pool_2x2(x):
    outputs=tf.nn.max_pool2d(input=x,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
    return outputs

tf.reset_default_graph()
# 输入层
xs = tf.placeholder(tf.float32, [None, 16])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 16, 1, 1])
# 卷积层1 激活 池化
W_conv1 = weight_variable([7, 1, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1)) #选取Relu作为激活函数
h_pool1 = max_pool_2x2(h_conv1)
# h_pool1.shape 8*1*1*32
# 卷积层2 激活 池化
W_conv2 = weight_variable([5, 1, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2))
h_pool2 = max_pool_2x2(h_conv2)
# h_pool2.shape 4*1*1*64
# 全连接层1 激活 dropout
h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 1 * 1 * 64])
W_fc1 = weight_variable([4 * 1 * 1 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 全连接层2 softmax
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 选取交叉熵作为损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)),
                                              reduction_indices=[1]))
# 使用梯度下降算法
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


def generate_batch(x, y, batch_size): # 训练集train按batch大小传输进神经网络模型中
    batch_xs = []
    batch_ys = []
    for batch_i in range(x.shape[0] // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs.append(x[start:end])
        batch_ys.append(y[start:end])
    return batch_xs, batch_ys  # 生成每一个batch



x,y = get_datacsv('data/x.csv', 'data/y.csv')

plt.figure(figsize=(12, 6))  # 创建绘图对象
repeat_count=10
repeat_accuracy = 0
repeat_epochs = []
repeat_train_Acc = []
repeat_test_Acc= []
for repeat_i in range(repeat_count):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)  # 训练集:测试集=8:2,按照y中的比例分配
    batch_xs, batch_ys = generate_batch(x_train, y_train, batch_size)
    epochs=[]
    train_Acc = []
    test_Acc = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 0
        while (epoch <= max_epoch):
            for i in range(len(batch_xs)):
                # 训练集train每次传输进神经网络模型中batch大小的数据
                sess.run(train_step, feed_dict={xs: batch_xs[i], ys: batch_ys[i], keep_prob: keep_prob_rate})
                epoch += 1
                acc = compute_accuracy(x_train[:1000], y_train[:1000])  # 训练集train的准确率
                accuracy = compute_accuracy(x_test, y_test)  # 测试机test的准确率
                epochs.append(epoch)
                test_Acc.append(acc)
                train_Acc.append(accuracy)
                print('repeat', repeat_i, ' epoch: ', epoch, ' train accuracy: ', acc)
                print('repeat', repeat_i, ' epoch: ', epoch, ' test accuracy: ', accuracy)

        #经过 max_epoch 轮迭代后的模型对测试集test的准确率
        accuracy = compute_accuracy(x_test, y_test)
        repeat_accuracy+=accuracy
        repeat_epochs.append(epochs)
        repeat_train_Acc.append(train_Acc)
        repeat_test_Acc.append(test_Acc)


repeat_epochs=np.array(repeat_epochs).mean(axis=0) #交叉验证取平均值
repeat_train_Acc=np.array(repeat_train_Acc).mean(axis=0)
repeat_test_Acc=np.array(repeat_test_Acc).mean(axis=0)
print('----------------------------------------------------')
repeat_accuracy/=repeat_count
print('Finally Test Accuracy: ', accuracy)

# 可视化
plt.plot(repeat_epochs, repeat_train_Acc, color='g', marker='o', linestyle='--', markersize=3, linewidth=1, label='train Accuracy')
plt.plot(repeat_epochs, repeat_test_Acc, color='c', marker='o', linestyle='--', markersize=3, linewidth=1,label='test Accuracy')

#x_major_locator = MultipleLocator(500)
#y_major_locator = MultipleLocator(0.05)
#ax = plt.gca()  # ax为两条坐标轴的实例
#ax.xaxis.set_major_locator(x_major_locator)  # x轴的主刻度设置
#ax.yaxis.set_major_locator(y_major_locator)  # y轴的主刻度设置
plt.xlabel("epoch", fontsize='15')  # X轴标签
plt.ylabel("Accuracy", fontsize='15')  # Y轴标签
plt.legend(fontsize='15', loc='upper right')


plt.savefig("data/epoch_Accuracy.png")  # 保存图
plt.show()  # 显示图