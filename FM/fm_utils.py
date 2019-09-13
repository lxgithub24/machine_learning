# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 13:26
# @Author  : RIO
# @desc: 公共方法
import numpy as np


def generator_nonzero(sample):
    """
    generate the nonzero index in the feature
    :param sample:  one sample          vector
    :return: generator
    """
    for j in range(len(sample)):
        if sample[j] != 0:
            yield j


def cal_loss(true_label, probability):
    """
    calculate the log_loss between ground true-label and prediction
    :param true_label: the ground truth label for the sample	{0, 1}
    :param probability: the prediction of the trained model		[0, 1]
    :return: logloss
    """
    probability = max(min(probability, 1. - 1e-15), 1e-15)
    return -np.log(probability) if true_label == 1 else - \
        np.log(1 - probability)


def cal_loss2(true_label, probability):
    """
    calculate the softmax log_loss between ground true-label and prediction for one single sample
    note: the probability has been normalized (no need to max or min operation)
    :param true_label: the ground truth label vector for the sample         -array
    :param probability: the prediction vector of the trained model          -array
    :return: logloss
    """
    k = np.argmax(true_label)
    return -np.log(probability[k])


def evaluate_model(preds, labels):
    """
    evaluate the model errors on a set of data (not one single sample)
    :param preds: the prediction of unseen samples          (n_sample, n_label)
    :param labels: the ground truth labels                  (n_sample, n_label)
    :return:
    """
    shapes = len(labels.shape)
    if shapes == 2:
        # multi-class classification-find the max-index per row

        max_index = np.argmax(preds, axis=1)
        for i, p in enumerate(max_index):
            preds[i, p] = 1
        preds[preds < 1.] = 0
    else:
        # binary classification-default (n_sample, )
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
    return np.abs(preds - labels).sum() / (len(labels) * shapes) * 100


def get_auc(scores, labels):
    """
    calculate the auc indicator on a set of data
    :param scores: the probability of each sample   [0, 1]-array
    :param labels: the ground truth labels          {0, 1}-array
    :return: auc indicator
    """
    data_shape = labels.shape
    pos_num = np.sum(labels, axis=0)
    neg_num = len(labels) - pos_num
    # rank scores
    rank_index = np.argsort(scores, axis=0, kind='quicksort')
    if len(data_shape) == 1:
        rank_sum = 0.0
        for i in range(data_shape[0]):
            if labels[rank_index[i]] == 1:
                rank_sum += (i + 1)
        # calculate the auc
        denominator = pos_num * neg_num
        if denominator == 0:
            res = 0
        else:
            res = (rank_sum - 0.5 * (pos_num + 1) * pos_num) / denominator

    else:
        rank_sum = np.zeros(data_shape[1])
        res = 0.0
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                if labels[rank_index[i, j], j] == 1:
                    rank_sum[j] += (i + 1)
        # calculate the auc
        denominator = pos_num * neg_num
        for j in range(data_shape[1]):
            if denominator[j] == 0:
                res += 0.0
            else:
                numerator = rank_sum[j] - 0.5 * (pos_num[j] + 1) * pos_num[j]
                res += numerator / denominator[j]
        res = res / data_shape[1]
    return res


def logistic0(var):
    """
    calculate the logistic value of one variable
    :param var: the input variable
    :return: logistic value
    """
    var = max(min(var, 100), -100)
    return 1. / (1 + np.exp(-var))


def logistic(var):
    """
    extend to multi-dimension ndarray   (1,2,3,4)multi-dimensions
    :param var: float/int/ndarray
    :return:
    """
    if isinstance(var, np.ndarray):
        shapes = var.shape
        length = np.multiply.reduce(shapes)
        var = np.reshape(var, length)
        res = np.zeros(length)
        for i in range(length):
            res[i] = logistic0(var[i])
        res = np.reshape(res, shapes)
    else:
        res = logistic0(var)
    return res


def softmax(var):
    """
    calculate the softmax value of one vector variable
    :param var: the input vector
    :return: softmax vector
    """
    e_x = np.exp(var - np.max(var))
    output = e_x / e_x.sum()
    return output


def generate_samples(dimension, n_samples):
    """
    generate samples according to the user-defined requirements
    :param dimension:
    :param n_samples:
    :return:
    """
    samples = np.random.rand(n_samples, dimension)
    labels = np.random.randint(0, 2, (n_samples,))
    return samples, labels
