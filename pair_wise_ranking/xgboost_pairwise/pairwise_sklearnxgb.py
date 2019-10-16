# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 20:39
# @Author  : RIO
import xgboost as xgb
from pair_wise_ranking.xgboost_pairwise.data_process import get_model_data
import os
import pickle

# 当前工作地址
curr_dir = os.path.dirname(__file__)
x_train, y_train, group_train, x_valid, y_valid, group_valid, x_test, y_test, group_test = get_model_data()


def train():
    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 6, 'n_estimators': 4}
    model = xgb.sklearn.XGBRanker(**params)
    model.fit(x_train, y_train, group_train,
              eval_set=[(x_valid, y_valid)], eval_group=[group_valid])
    with open(curr_dir + './data/data_model/pairwise_sklearnxgb.model', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    return 1


def predict():
    model = pickle.load(curr_dir + './data/data_model/pairwise_sklearnxgb.model')
    pred = model.predict(x_test)
    return pred


if __name__ == '__main__':
    # train()
    predict()
