# -*- coding: utf-8 -*-
# @Time    : 2017/1/10 9:12
# @Author  : RIO
# @desc: 召回部分
import pandas as pd
df = pd.read_csv('./a.txt', quoting=3)
print('question' in df)