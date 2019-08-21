# -*- coding: utf-8 -*-
# @Time    : 2019/8/13 16:14
# @Author  : RIO
# @desc: TODO:DESC

import numpy as np
from sklearn import svm
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt


def extend(a, b, r):
    x = a - b
    m = (a + b) / 2
    return m - r * x / 2, m + r * x / 2


if __name__ == '__main__':
    np.random.seed(0)
    N = 20
    x = np.empty((4 * N, 2))
    # print(x)
    means = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    sigmas = [np.eye(2), 2 * np.eye(2), np.diag((1, 2)), np.array(((2, 1), (1, 2)))]
    for i in range(4):
        mn = stats.multivariate_normal(means[i], sigmas[i] * 0.3)
        x[i * N:(i + 1) * N, :] = mn.rvs(N)
    a = np.array((0, 1, 2, 3)).reshape((-1, 1))
    y = np.tile(a, N).flatten()
    clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')
    # clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovo')
    clf.fit(x, y)
    y_hat = clf.predict(x)
    acc = accuracy_score(y, y_hat)
    np.set_printoptions(suppress=True)
    print('True prediction numbers:%d, predictions:%.3f%%' % (round(acc * 4 * N), 100 * acc))
    print('decision_function:\n', clf.decision_function(x))
    print('y_hat:\n', y_hat)
    # 开始画图
    x1_min, x2_min = np.min(x, axis=0)
    x1_max, x2_max = np.max(x, axis=0)
    x1_min, x1_max = extend(x1_min, x1_max, 1.05)
    x2_min, x2_max = extend(x2_min, x2_max, 1.05)
    # 生成网格采样点
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    y_test = clf.predict(x_test)
    y_test = y_test.reshape(x1.shape)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_test, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=cm_dark, alpha=0.7)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)
    plt.tight_layout(pad=2.5)
    plt.title('SVM Metholds--one/one or one/other', fontsize=18)
    plt.show()