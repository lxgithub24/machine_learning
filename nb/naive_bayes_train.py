# -*- coding: utf-8 -*-
# @Time    : 2019/6/5 15:34
# @Author  : RIO
# @desc: 朴素贝叶斯

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from algorithm.nb.data_process import read_csv, write_csv

tfmodel = TfidfVectorizer(use_idf=True)
data_frame = read_csv('./data/train_data.csv')
train_X, test_X, train_y, test_y = train_test_split(data_frame['processed_data'], data_frame['label'], test_size=0.2)
tfidf_train_data = tfmodel.fit_transform(train_X)
tfidf_test_data = tfmodel.transform(test_X)
multi_nomial_model = MultinomialNB(alpha=1e-10)
multi_nomial_model.fit(tfidf_train_data, train_y)
predicted = multi_nomial_model.predict(tfidf_test_data)
predict_prob = multi_nomial_model.predict_proba(tfidf_test_data)
# print(test_X)
# print([list(test_y), list(predicted)])
print(metrics.classification_report(test_y, predicted))  # 输出分类信息
print('################################################')
label = list(set(train_y))  # 去重复，得到标签类别
print(metrics.confusion_matrix(test_y, predicted, labels=label).ravel())  # 输出混淆矩阵信息
# data_frame['predict_label'] = predicted
# write_csv('./data/res_data.csv', data_frame)
