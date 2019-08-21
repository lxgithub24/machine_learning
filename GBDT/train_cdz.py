# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train_cdz
   Description :
   Author :       DrZ
   date：          2018/8/25
-------------------------------------------------
   Change Activity:
                   2018/8/25:
-------------------------------------------------
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib


data = pd.read_csv(r"./data/data_train.csv")
x_columns = []
for x in data.columns:
    if x not in ['id', 'label']:
        x_columns.append(x)
X = data[x_columns]
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(X, y)

# 模型训练，使用GBDT算法
gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(x_train, y_train.ravel())
joblib.dump(gbr, './data/train_model_result4.m')   # 保存模型

y_gbr = gbr.predict(x_train)
y_gbr1 = gbr.predict(x_test)
acc_train = gbr.score(x_train, y_train)
acc_test = gbr.score(x_test, y_test)
print(acc_train)
print(acc_test)

