# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 11:26
# @Author  : RIO
# @desc: TODO:DESC
import tensorflow as tf

# [5,1]的矩阵，5组数据，每组数据为1个。
# tf.layers.dense会根据这个shape，自动调整输入层单元数。
input = tf.ones([5, 2])
print('-'*50)
output = tf.layers.dense(input, 5)
s = tf.Session()
s.run(tf.global_variables_initializer())
print(s.run(input))
print(s.run(output))

# print(output)
# print('-'*50)
# input = tf.ones([3, 2])
# output = tf.layers.dense(input, 10)
# print(output.get_shape())
#
# input = tf.ones([1, 7, 20])
# output = tf.layers.dense(input, 10)
# print(output.get_shape())
#
# input = tf.ones([1, 7, 11, 20])
# output = tf.layers.dense(input, 10)
# print(output.get_shape())