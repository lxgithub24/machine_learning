# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 13:23
# @Author  : RIO
# @desc: 初始化數據集

# ================基于XGBoost原生接口的分类=============
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   # 准确率
# 加载样本数据集
iris = load_iris()
X,y = iris.data,iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565) # 数据集分割