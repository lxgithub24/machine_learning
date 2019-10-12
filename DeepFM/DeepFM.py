# -*- coding: utf-8 -*-
# @Time    : 2019/9/12 18:43
# @Author  : RIO
# @desc: 文章地址：https://www.cnblogs.com/wkang/p/9881921.html， 参考：https://github.com/ChenglongChen/tensorflow-DeepFM
import tensorflow as tf
import numpy as np
import sys
import os

BASE_PATH = os.path.dirname(os.path.dirname(__file__))


class Args():
    feature_num = 100
    field_num = 15
    hidden_size = 256
    dim_of_layer = [512, 256, 128]
    epoch = 3
    batch_num = 64
    learning_rate = 1.0
    l2_regularizer = 0.01
    checkpoint_dir = os.path.join(BASE_PATH, 'data/saver/ckpt')
    is_training = True


class DeepFM():
    def __init__(self, args):
        self.batch_num = args.batch_num
        self.feature_num = args.feature_num
        self.hidden_size = args.hidden_size
        self.weight = dict()
        self.field_num = args.field_num
        self.dim_of_layer = args.dim_of_layer
        self.l2_regularizer = args.l2_regularizer
        self.learning_rate = args.learning_rate
        self.init_graph()
        pass

    def init_graph(self):
        # 初始化喂入参数
        self.feature_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
        self.feature_value = tf.placeholder(tf.int32, shape=[None, None], name='feature_value')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        # -------------------------------------------------FM部分初始化----------------------------------------------------------
        # FM部分
        # w:一次项系数
        self.weight['fm_single_feature'] = tf.Variable(
            initial_value=tf.random_normal(shape=[self.feature_num, 1], mean=0.0, stddev=1.0), name='fm_single_feature')
        # v:交叉项系数
        self.weight['fm_intersection_feature'] = tf.Variable(
            initial_value=tf.random_normal(shape=[self.feature_num, self.hidden_size], mean=0.0, stddev=0.01),
            name='fm_intersection_feature')
        # -------------------------------------------------一次项---------------------------------------------------------------
        # 一次项:(wx,wx,wx,...)
        embedding_value = tf.nn.embedding_lookup(self.weight['fm_single_feature'], self.feature_index)
        # 两个乘子分别是w:一行，field列；x：len(sample)行，field列
        embedding_value = tf.multiply(embedding_value, tf.reshape(self.feature_value),
                                      [-1, self.field_num, 1])
        # reduce_sum得到的是len(sample)长度，每个值代表每个样本的wx和。
        fm_single_value = tf.reduce_sum(embedding_value, 2)
        # -------------------------------------------------一次项---------------------------------------------------------------
        # -------------------------------------------------二次项---------------------------------------------------------------
        # 二次项：ab=1/2((a+b)^2-a^2-b^2)
        # 查表获取X
        fm_intersection_value = tf.nn.embedding_lookup(self.weight['fm_intersection_feature'], self.feature_index)
        # v*x，得到向量(vx,vx,vx,...)
        fm_intersection_value = tf.multiply(fm_intersection_value, tf.reshape(self.feature_value),
                                            [-1, self.field_num, 1])
        # 计算向量平方和
        fm_intersection_value_sum_of_square = tf.reduce_sum(tf.square(fm_intersection_value), axis=1)
        # 计算向量和的平方
        fm_intersection_value_square_of_sum = tf.square(tf.reduce_sum(fm_intersection_value, axis=1))
        # 计算fm二次项value
        fm_intersection_value = 0.5 * tf.subtract(
            fm_intersection_value_square_of_sum - fm_intersection_value_sum_of_square)
        # fm部分最终结果
        fm_part = tf.concat([fm_single_value, fm_intersection_value], axis=1)
        # -------------------------------------------------二次项---------------------------------------------------------------
        # -------------------------------------------------deep部分-------------------------------------------------------------
        # deep部分初始化
        # 深度
        layer_deep = len(self.dim_of_layer)
        # 初始化embedding的大小
        input_embedding_size = self.field_num * self.hidden_size
        # 第一层的特征维度
        sigma_of_layer_0 = np.square(2 / (input_embedding_size + self.dim_of_layer[0]))
        # 初始化0层的w
        self.weight['layer_0'] = tf.Variable(
            tf.random_normal([input_embedding_size, self.dim_of_layer[0]], 0.0, sigma_of_layer_0))
        # 初始化0层的b
        self.weight['bias_0'] = tf.Variable(tf.random_normal([1, self.dim_of_layer[0]], 0.0, sigma_of_layer_0))
        # deep中1到n层的w和b
        for i in range(1, layer_deep):
            curr_embedding_size = self.dim_of_layer[i - 1]
            sigma_of_curr_layer = np.square(2 / (curr_embedding_size + self.dim_of_layer[i]))
            self.weight['layer_{}'.format(i)] = tf.Variable(
                tf.random_normal([curr_embedding_size, self.dim_of_layer[i]], 0.0, sigma_of_curr_layer))
        # 初始化最后一层的w。因为最后一层用户计算fm+deep的concatenate，所以embedding的长度为：field_num（fm一次项长度）+embedding_size(隐层即fm中v的长度)+self.dim_of_layer[-1]（最后一层的向量长度）
        output_size = self.field_num + self.hidden_size + self.dim_of_layer[-1]
        sigma_of_lastlayer = np.sqrt(2 / (output_size + 1))
        self.weight['last_layer'] = tf.Variable(tf.random_normal([output_size, 1], 0.0, sigma_of_lastlayer))
        self.weight['last_bias'] = tf.Variable(tf.constant(0.01))

        # deep部分计算
        deep_part = tf.reshape(embedding_value, [-1, self.field_num * self.hidden_size])
        for i in range(len(self.dim_of_layer)):
            # y = wx+b
            deep_part = tf.add(tf.multiply(deep_part, self.weight['layer_{}'.format(i)]),
                               self.weight['bias_{}'.format(i)])
            # y = relu(y)
            deep_part = tf.nn.relu(deep_part)

        # -------------------------------------------------deep部分结束----------------------------------------------------------
        # -------------------------------------------------deepFM部分-------------------------------------------------------------
        # 获取最后一层词向量
        self.output = tf.concat([fm_part, deep_part], axis=1)
        # y = w*x+b
        self.output = tf.add(tf.multiply(self.weight['last_layer'], self.output), self.weight['last_bias'])
        self.output = tf.nn.sigmoid(self.output)
        # -------------------------------------------------deepFM部分-------------------------------------------------------------
        # 计算loss：loss = tf.losses.log_loss(label, out)
        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.output + 1e-24) + (1 - self.label) * tf.log(1 - self.output + 1e-24))
        # 增加正则项:sum(w^2)/2 * l2_reg_rate
        for i in range(len(self.dim_of_layer)):
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_regularizer)(self.weight['layer_{}'.format(i)])
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_regularizer)(self.weight['last_layer'])
        # 获得梯度下降优化器
        gradient_descent_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, trainable=False)
        train_var = tf.trainable_variables()
        clip_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_var), 5)
        self.gradient_descent = gradient_descent_optimizer.apply_gradients(zip(clip_gradients, train_var),
                                                                           global_step=self.global_step)

    def train(self, session, feature_index, feature_value, label):
        loss, _, step = session.run([self.loss, self.gradient_descent, self.global_step],
                                    feed_dict={self.feature_index: feature_index, self.feature_value: feature_value,
                                               self.label: label})
        return loss, step

    def predict(self, session, feature_index, feature_value):
        result = session.run([self.output],
                             feed_dict={self.feature_index: feature_index, self.feature_value: feature_value})
        return result

    def save(self, session, path):
        tf.train.Saver().save(session, path)

    def restore(self, session, path):
        tf.train.Saver().restore(session, path)


# 获取每批数据
def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], np.array(y[start:end])


# 运行图
def run_graph():
    args = Args()
    data = {}
    # 可以增加gpu使用的设置
    with tf.Session() as session:
        deepFM = DeepFM(args)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        count = int(len(data['label']) / args.batch_num)
        sys.stdout.flush()
        if args.is_training:
            for i in range(args.epoch):
                for j in range(count):
                    X_index, X_value, y = get_batch(data['xi'], data['xv'], data['label'], args.batch_num, j)
                    loss, step = deepFM.train(session, X_index, X_value, y)
                    if j % 100 == 0:
                        deepFM.save(session, args.checkpoint_dir)
        else:
            deepFM.restore(session, args.checkpoint_dir)
            for j in range(0, count):
                X_index, X_value, y = get_batch(data['xi'], data['xv'], data['label'], args.batch_num, j)
                result = deepFM.predict(session, X_index, X_value)
                print(result)


if __name__ == '__main__':
    run_graph()
