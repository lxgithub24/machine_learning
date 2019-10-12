# -*- coding: utf-8 -*-
# @Time    : 2019/9/5 11:57
# @Author  : RIO
# @desc: TODO:DESC
import numpy as np
import pandas as pd
from sklearn.externals import joblib


# 加载模型并预测
gbr = joblib.load('train_model_result4.m')    # 加载模型
test_data = pd.read_csv(r"./data/data_test.csv")
testx_columns = []
for xx in test_data.columns:
    if xx not in ['id', 'label']:
        testx_columns.append(xx)
test_x = test_data[testx_columns]
test_y = gbr.predict(test_x)
test_y = np.reshape(test_y, (36644, 1))

# 保存预测结果
df = pd.DataFrame()
df['id'] = test_data['id']
df['label'] = test_y
df.to_csv("./data_predict.csv", header=None, index=None)
