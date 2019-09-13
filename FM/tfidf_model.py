# -*- coding: utf-8 -*-
# @Time    : 2019/9/7 12:49
# @Author  : RIO
# @desc: TODO:DESC
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from algorithm.FM.misc import get_origin_train_data, read_txt, dump_data

# 当前工作地址
curr_dir = os.path.dirname(__file__)


def get_X_y():
    # 加载训练数据
    X, y = get_origin_train_data()
    return X, y


def get_2and3ngram_tfidf_model(X, y):
    # 加载停用词
    stop_words = read_txt(curr_dir + '/../data/global/stop_words.txt')
    # 构建2ngram tfidf模型
    tfidf_1ngram_model = TfidfVectorizer(stop_words=stop_words, max_df=0.6, sublinear_tf=True, ngram_range=(1, 1),
                                         max_features=10000)
    tfidf_1ngram_X = tfidf_1ngram_model.fit_transform(X)
    # 构建2ngram tfidf模型
    tfidf_2ngram_model = TfidfVectorizer(stop_words=stop_words, max_df=0.6, sublinear_tf=True, ngram_range=(1, 2),
                                         max_features=20000)
    tfidf_2ngram_X = tfidf_2ngram_model.fit_transform(X)

    # 构建3ngram tfidf模型
    tfidf_3ngram_model = TfidfVectorizer(stop_words=stop_words, max_df=0.6, sublinear_tf=True, ngram_range=(1, 3),
                                         max_features=30000)
    tfidf_3ngram_X = tfidf_3ngram_model.fit_transform(X)
    print(tfidf_3ngram_X.shape)
    dump_dict = {'1ngram': tfidf_1ngram_X, '1ngram_model': tfidf_1ngram_model, '2ngram': tfidf_2ngram_X, '2ngram_model': tfidf_2ngram_model, '3ngram_model': tfidf_3ngram_model,
                 '3ngram': tfidf_3ngram_X, 'label': y}
    # dump_dict = {'1ngram': tfidf_1ngram_X, '1ngram_model': tfidf_1ngram_model, 'label': y}
    dump_data(dump_dict, curr_dir + '/../data/FM/tfidf_2and3ngram_model.model')


if __name__ == '__main__':
    X, y = get_X_y()
    get_2and3ngram_tfidf_model(X, y)

