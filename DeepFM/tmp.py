# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 12:08
# @Author  : RIO
# @desc: TODO:DESC
import numpy as np
# import tensorflow as tf
a = np.array([[1,2],[3,4]])
b = np.array([[3,4]])
# c = tf.multiply(a,b)
# with tf.Session as session:
#     session.run([c])
# print(c.eval())
c = np.multiply(a,b)

print(c)