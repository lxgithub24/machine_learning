# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 14:25
# @Author  : RIO
# @desc: TODO:DESC
from algorithm.FM.FactorizationMachine import FM
from algorithm.FM.misc import load_data, dump_data
import os

# 当前工作地址
curr_dir = os.path.dirname(__file__)


# 模型训练
def train_fm_model(iteration_=20):
    # define the hyper-parameter
    alpha_w_, alpha_v_, beta_w_, beta_v_ = 0.2, 0.2, 0.2, 0.2
    lambda_w1_, lambda_w2_, lambda_v1_, lambda_v2_ = 0.2, 0.2, 0.2, 0.2
    hiddens = 8
    sigma_ = 1.0
    # iteration_ = 20

    # 加载tfidf模型
    tfidf_model = load_data(curr_dir + '/../data/FM/tfidf_2and3ngram_model.model')
    X = tfidf_model.get('1ngram').A
    y = tfidf_model.get('label')
    # create the fm model
    fm_model = FM(dim=X.shape[1], dim_lat=hiddens, sigma=sigma_, alpha_w=alpha_w_, alpha_v=alpha_v_, beta_w=beta_w_,
            beta_v=beta_v_, lambda_w1=lambda_w1_, lambda_w2=lambda_w2_, lambda_v1=lambda_v1_, lambda_v2=lambda_v2_)

    fm_model.train_ftrl(X, y, iteration=iteration_)
    dump_data(fm_model, curr_dir + '/../data/FM/fm_model.model')


if __name__ == '__main__':
    train_fm_model(2)
