# -*- coding: utf-8 -*-
# @Time    : 2019/6/6 13:38
# @Author  : RIO
# @desc: 线性回归训练
from sklearn.linear_model import LinearRegression


from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
import numpy as np
model = linear_model.LinearRegression()
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
model.fit(X, y)
print(model.intercept_)  #截距
print(model.coef_)  #线性模型的系数
a = model.predict([[12]])
# a[0][0]
print("预测一张12英寸匹萨价格：{:.2f}".format(model.predict([[12]])[0][0]))