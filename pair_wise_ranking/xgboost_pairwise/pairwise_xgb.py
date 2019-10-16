# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 20:39
# @Author  : RIO
import xgboost as xgb
from xgboost import DMatrix
from pair_wise_ranking.xgboost_pairwise.data_process import get_model_data
import os
import pickle

# 当前工作地址
curr_dir = os.path.dirname(__file__)
x_train, y_train, group_train, x_valid, y_valid, group_valid, x_test, y_test, group_test = get_model_data()


def train():
    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test)

    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)

    params = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0,
              'min_child_weight': 0.1, 'max_depth': 6}
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4,
                          evals=[(valid_dmatrix, 'validation')])
    with open(curr_dir + './data/data_model/pairwise_origin_version.model', 'wb') as f:
        pickle.dump(xgb_model, f, pickle.HIGHEST_PROTOCOL)
    return 1


def predict():
    xgb_model = pickle.load(curr_dir + './data/data_model/pairwise_origin_version.model')
    pred = xgb_model.predict(x_test)
    print(pred)


if __name__ == '__main__':
    # train()
    predict()
