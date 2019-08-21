# -*- coding: utf-8 -*-
# @Time    : 2019/8/10 13:23
# @Author  : RIO
# @desc: TODO:DESC
import pandas as pd

features = pd.read_csv('temps.csv')
features.head(5)
features = pd.get_dummies(features)
features.head(5)

#靶向量（因变量）
targets = features['actual']

# 从特征矩阵中移除actual这一列
#axis=1表示移除列的方向是列方向
features= features.drop('actual', axis = 1)

# 特征名列表
feature_list = list(features.columns)

from sklearn.model_selection import train_test_split

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size = 0.25, random_state = 42)

import numpy as np

#选中test_features所有行
#选中test_features中average列
baseline_preds = test_features.loc[:, 'average']

baseline_errors = abs(baseline_preds - test_targets)
print('平均误差: ', round(np.mean(baseline_errors), 2))

from sklearn.ensemble import RandomForestRegressor

#1000个决策树
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)
rf.fit(train_features, train_targets)

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
           oob_score=False, random_state=42, verbose=0, warm_start=False)


predictions = rf.predict(test_features)

errors = abs(predictions - test_targets)

print('平均误差:', round(np.mean(errors), 2))


#计算平均绝对百分误差mean absolute percentage error (MAPE)
mape = 100 * (errors / test_targets)

accuracy = 100 - np.mean(mape)
print('准确率:', round(accuracy, 2), '%.')

print('模型中的决策树有', len(rf.estimators_), '个')


#从1000个决策树中抽选出前5个看看
print(rf.estimators_[:5])

from sklearn.tree import export_graphviz
import pydot

# 从这1000个决策树中，就选第6个决策树吧。
tree = rf.estimators_[5]

#将决策树输出到dot文件中
export_graphviz(tree,
                out_file = 'tree.dot',
                feature_names = feature_list,
                rounded = True,
                precision = 1)

# 将dot文件转化为图结构
(graph, ) = pydot.graph_from_dot_file('tree.dot')

#将graph图输出为png图片文件
graph.write_png('tree.png')

print('该决策树的最大深度（层数）是:', tree.tree_.max_depth)

rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
rf_small.fit(train_features, train_labels)

tree_small = rf_small.estimators_[5]

export_graphviz(tree_small, out_file = 'small_tree.dot',
                feature_names = feature_list,
                rounded = True,
                precision = 1)

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

graph.write_png('small_tree.png')

#获得特征重要性信息
importances = list(rf.feature_importances_)

feature_importances = [(feature, round(importance, 2))
                       for feature, importance in zip(feature_list, importances)]

#重要性从高到低排序
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

import matplotlib.pyplot as plt


#设置画布风格
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')