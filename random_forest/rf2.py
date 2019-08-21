# -*- coding: utf-8 -*-
# @Time    : 2019/8/10 13:32
# @Author  : RIO
# @desc: TODO:DESC
import pydot
from sklearn.tree import export_graphviz
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

# Read data
# DataFrame
train_df = pd.read_csv('./data/train.csv')
# Cnvert DataFrame to array
train_data = train_df.values

test_df = pd.read_csv('./data/test.csv')
test_data = test_df.values
test_df.head(5)
# 特征名列表
feature_list = list(test_df.columns)
# 画图
plt.figure(figsize=(12, 8))
sns.countplot(x='label', data=train_df)
plt.title('Distribution of Numbers')
plt.xlabel('Numbers')

num_features = train_data.shape[0]  # 这里返回的是train_data的行数作为特征个数
print("Number of all features: \t\t", num_features)
split = int(num_features * 2 / 3)  # 这里是取2/3行也就是前28000行作为训练 后1/3也就是14000作为测试

train = train_data[:split]  # 取出前28000行作为训练数据
test = train_data[split:]  # 取出后14000行作为测试数据

print("Number of features used for training: \t", len(train), "\nNumber of features used for testing: \t", len(test))

# 开始使用随机森林分类器
clf = RandomForestClassifier(n_estimators=100)  # 定义为100 tree

# 开始训练，训练的X数据格式为[[]]，训练的y值为[]也就是经过ravel后的数据
# 如果你问我ravel()的作用是什么，就是不管什么数据格式的数据都转成一个array，这样每个元素都是一个平等且顺序的位置
model = clf.fit(train[:, 1:], train[:, 0].ravel())

# 然后预测
output = model.predict(test[:, 1:])
# 计算准确度：将每个
acc = np.mean(output == test[:, 0].ravel()) * 100
print("The accuracy of the pure RandomForest classifier is: \t", acc, "%")

# 利用
clf = RandomForestClassifier(n_estimators=100)  # 100 trees

# 用全部训练数据来做训练
target = train_data[:, 0].ravel()
train = train_data[:, 1:]
model = clf.fit(train, target)

output = model.predict(test_data)

pd.DataFrame({"ImageId": range(1, len(output) + 1), "Label": output}).to_csv('./data/out.csv', index=False, header=True)

print('-' * 20)

# 从100个决策树中抽选出前5个看看
print(clf.estimators_[:5])


# 从这1000个决策树中，就选第6个决策树吧。
tree = clf.estimators_[5]

# 将决策树输出到dot文件中
export_graphviz(tree, out_file='./data/tree.dot', feature_names=feature_list, rounded=True, precision=1)

# 将dot文件转化为图结构
(graph,) = pydot.graph_from_dot_file('./data/tree.dot')

# 将graph图输出为png图片文件
graph.write_png('tree.png')

print('该决策树的最大深度（层数）是:', tree.tree_.max_depth)
print('-' * 20)

# rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
# rf_small.fit(train_features, train_labels)
#
# tree_small = rf_small.estimators_[5]
#
# export_graphviz(tree_small, out_file = 'small_tree.dot',
#                 feature_names = feature_list,
#                 rounded = True,
#                 precision = 1)
#
# (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
#
# graph.write_png('small_tree.png')
#
# #获得特征重要性信息
# importances = list(rf.feature_importances_)
#
# feature_importances = [(feature, round(importance, 2))
#                        for feature, importance in zip(feature_list, importances)]
#
# #重要性从高到低排序
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#
# # Print out the feature and importances
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#
# import matplotlib.pyplot as plt
#
#
# #设置画布风格
# plt.style.use('fivethirtyeight')
#
# # list of x locations for plotting
# x_values = list(range(len(importances)))
#
# # Make a bar chart
# plt.bar(x_values, importances, orientation = 'vertical')
#
# # Tick labels for x axis
# plt.xticks(x_values, feature_list, rotation='vertical')
#
# # Axis labels and title
# plt.ylabel('Importance')
# plt.xlabel('Variable')
# plt.title('Variable Importances')
#
# print('-'* 20)
