# -*- coding: utf-8 -*-
# @Time    : 2019/6/5 15:54
# @Author  : RIO
# @desc: 数据处理
import pandas as pd
import jieba
import re
jieba.load_userdict('./data/user_dict.txt')
tfidf_corpus = []
stop_sign = re.compile("[^\u4e00-\u9fa5a-z0-9.]")


def read_csv(_dir):
    data_frame = pd.read_csv(_dir)
    for i in data_frame['description']:
        tfidf_corpus.append(' '.join(jieba.lcut(stop_sign.sub('', i))))
    data_frame['processed_data'] = tfidf_corpus
    return data_frame

def write_csv(_dir, data):
    df = pd.DataFrame(data=data, columns=['description', 'label', 'processed_data', 'predict_label']).to_csv(_dir, index=False)
