# -*- coding: utf-8 -*-
# @Time    : 2019/9/9 16:45
# @Author  : RIO
# @desc: FM模型的预测
import os
from algorithm.FM.misc import load_data, dump_data, get_origin_train_data
from algorithm.FM.fm_model import train_fm_model

# 当前工作地址
curr_dir = os.path.dirname(__file__)


# 讀取測試數據
def get_X_y():
    X, y = get_origin_train_data(curr_dir + '/../data/predict/test_data.csv')
    return X, y


# 預測
def predict(X, y):
    # 加載tfidf模型
    tfidf_model = load_data(curr_dir + '/../data/FM/tfidf_2and3ngram_model.model')
    predict_X = tfidf_model.get('1ngram_model').transform(X)
    # 加載FM模型
    fm_model = load_data(curr_dir + '/../data/FM/fm_model.model')
    predict_test = fm_model.predict(predict_X.A)
    tmp_predict_test = [str(i) for i in predict_test]
    with open(curr_dir + '/../data/predict/predict_label.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(list(tmp_predict_test)))
    dump_data(predict_test, curr_dir + '/../data/predict/predict_test.model')
    return predict_test


def main1():
    import time
    from algorithm.FM.tfidf_model import get_X_y as getxy, get_2and3ngram_tfidf_model
    # X, y = getxy()
    # get_2and3ngram_tfidf_model(X, y)
    t1 = time.time()
    # train_fm_model(iteration_=10)
    t2 = time.time()
    print('!t2###', t2 - t1)
    X, y = get_X_y()
    test_y = predict(X, y)
    t3 = time.time()
    print('!t3###', t3 - t2)


if __name__ == '__main__':
    main1()
