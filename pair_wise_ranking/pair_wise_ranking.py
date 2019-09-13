# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 20:39
# @Author  : RIO
# @desc: pair-wise ranking思想。转载：https://www.deeplearn.me/1982.html
import tensorflow as tf
import numpy as np

BATCH_SIZE = 100
y_train = []
X_train = []
Query = []
array_train_x1 = []
array_train_x0 = []

feature_num = 46
h1_num = 10


def extractFeatures(split):
    '''
    获取特征
    '''
    features = []
    for i in range(2, 48):
        features.append(float(split[i].split(':')[1]))
    return features


def extractQueryData(split):
    '''
    获取以下数据 quryid documentid 等
    Format:

    '''
    queryFeatures = [split[1].split(':')[1]]
    queryFeatures.append(split[50])
    queryFeatures.append(split[53])
    queryFeatures.append(split[56])

    return queryFeatures


def get_microsoft_data():
    '''
    获取基础样本特征数据
    :return:
    '''
    with open('/Users/leiyang/RankNet/Data/train.txt', 'r') as fp:
        for data in fp:
            split = data.split()
            y_train.append(int(split[0]))
            X_train.append(extractFeatures(split))
            Query.append(extractQueryData(split))


def get_pair_feature(y_train, Query):
    '''
    获取组合样本特征
    :return:
    '''
    pairs = []
    tmp_x0 = []
    tmp_x1 = []
    for i in range(0, len(Query)):
        for j in range(i + 1, len(Query)):
            # Only look at queries with the same id
            if (Query[i][0] != Query[j][0]):
                break
            # Document pairs found with different rating
            if (Query[i][0] == Query[j][0] and y_train[i] != y_train[j]):
                # Sort by saving the largest index in position 0
                if (y_train[i] > y_train[j]):
                    pairs.append([i, j])
                    tmp_x0.append(X_train[i])
                    tmp_x1.append(X_train[j])
                else:
                    pairs.append([j, i])
                    tmp_x0.append(X_train[j])
                    tmp_x1.append(X_train[i])

    array_train_x0 = np.array(tmp_x0)
    array_train_x1 = np.array(tmp_x1)
    print('Found %d document pairs' % (len(pairs)))
    return pairs, len(pairs), array_train_x0, array_train_x1


with tf.name_scope("input"):
    x1 = tf.placeholder(tf.float32, [None, feature_num], name="x1")
    x2 = tf.placeholder(tf.float32, [None, feature_num], name="x2")

# 添加隐层节点
with tf.name_scope("layer1"):
    with tf.name_scope("w1"):
        w1 = tf.Variable(tf.random_normal([feature_num, h1_num]), name="w1")
    with  tf.name_scope("b1"):
        b1 = tf.Variable(tf.random_normal([h1_num]), name="b1")

    # 此处没有添加激活函数
    with tf.name_scope("h1_o1"):
        h1_o1 = tf.matmul(x1, w1) + b1
        h1_o1 = tf.nn.relu(h1_o1)

    with tf.name_scope("h2_o1"):
        h1_o2 = tf.matmul(x2, w1) + b1
        h1_o2 = tf.nn.relu(h1_o2)

# 添加输出节点
with tf.name_scope("output"):
    with tf.name_scope("w2"):
        w2 = tf.Variable(tf.random_normal([h1_num, 1]), name="w2")

    with tf.name_scope("b2"):
        b2 = tf.Variable(tf.random_normal([1]))

    h2_o1 = tf.matmul(h1_o1, w2) + b2
    h2_o2 = tf.matmul(h1_o2, w2) + b2
    h2_o1 = tf.sigmoid(h2_o1)
    h2_o2 = tf.sigmoid(h2_o2)

# 根据输出节点计算概率值
with tf.name_scope("loss"):
    # o12 = o1 - o2
    h_o12 = h2_o1 - h2_o2
    pred = 1 / (1 + tf.exp(-h_o12))
    # 此处的 label_P 就是真实的概率，因为前面组 pair 数据已经人为将相关的样本放在
    # 前面，所以 Sij 均为 1，所以计算的结果就是 1
    lable_p = 1

    cross_entropy = -lable_p * tf.log(pred) - (1 - lable_p) * tf.log(1 - pred)

    reduce_sum = tf.reduce_sum(cross_entropy)
    loss = tf.reduce_mean(reduce_sum)

with tf.name_scope("train_op"):
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # step 1 解析 microsoft 数据集
    get_microsoft_data()
    # step 2 获取 pair 组合
    pairs, datasize, array_train_x0, array_train_x1 = get_pair_feature(y_train, Query)
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(0, 10000):
        start = (epoch * BATCH_SIZE) % datasize
        end = min(start + BATCH_SIZE, datasize)
        sess.run(train_op, feed_dict={x1: array_train_x0[start:end, :], x2: array_train_x1[start:end, :]})
        if epoch % 1000 == 0:
            l_v = sess.run(loss, feed_dict={x1: array_train_x0, x2: array_train_x1})

            result_0 = sess.run(h2_o1, feed_dict={x1: array_train_x0, x2: array_train_x1})
            result_1 = sess.run(h2_o2, feed_dict={x1: array_train_x0, x2: array_train_x1})
            # 使用所有的样本计算模型预测的准确率
            print(np.sum(result_0 > result_1) * 1.0 / datasize)
            # print  sess.run(cross_entropy,feed_dict={x1:array_train_x0, x2:array_train_x1})
            # print "------ epoch[%d] loss_v[%f] ------ "%(epoch, l_v)
