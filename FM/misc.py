# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 13:33
# @Author  : RIO
# @desc: TODO:DESC
import pickle
import os
import pandas as pd
import math
import numpy as np

# 当前工作地址
curr_dir = os.path.dirname(__file__)
# 最小值，防止log报错
__EPS = 1.4e-45
# 范数值单例
np_linalg_norm = {}
# w2v向量值单例
unitvec_sigle = {}


# 读取 X, y
def get_origin_train_data(_dir=curr_dir + '/../data/global/private_call_all_with_text.csv'):
    df = pd.read_csv(_dir)
    return df['content'], df['label']


# 读取txt文件
def read_txt(_dir):
    with open(_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [item.strip().replace('\r\n', '').replace('\n', '').replace('\n\r', '') for item in lines]
        return lines


def load_data(file_path):
    with open(file_path, "rb") as f:
        model = pickle.load(f, encoding='utf-8')
    return model


def dump_data(content, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)


# jsd相似度：https://www.zealseeker.com/archives/jensen-shannon-divergence-jsd-python/
def jsd(prob1, prob2):
    prob1_norm = sum(abs(p) for p in prob1)
    prob2_norm = sum(abs(p) for p in prob2)
    prob1 = [p / prob1_norm for p in prob1]
    prob2 = [p / prob2_norm for p in prob2]
    middle = [(prob1[idx] + prob2[idx]) / 2 for idx in range(len(prob1))]
    return 0.5 * (relative_entropy(prob1, middle) + relative_entropy(prob2, middle))


# KLD：kl散度
def relative_entropy(probx, proby):
    resultConEn = 0
    for i in range(len(probx)):
        resultConEn += probx[i] * math.log(max(probx[i] / max(proby[i], __EPS), __EPS))
    return resultConEn


# cosine计算相似度
def cosine(a, b):
    if np_linalg_norm.__contains__(str(a)):
        norm_a = np_linalg_norm[str(a)]
    else:
        norm_a = np.linalg.norm(a)
        np_linalg_norm[str(a)] = norm_a
    if np_linalg_norm.__contains__(str(b)):
        norm_b = np_linalg_norm[str(b)]
    else:
        norm_b = np.linalg.norm(b)
        np_linalg_norm[str(b)] = norm_b
    return a.dot(b.T) / (norm_a * norm_b)