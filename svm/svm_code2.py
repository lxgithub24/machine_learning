# -*- coding: utf-8 -*-
# @Time    : 2019/8/13 16:15
# @Author  : RIO
# @desc: TODO:DESC
import numpy as np
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors


def extend(a, b):
    return (1.01 * a - 0.01 * b), (1.01 * b - 0.01 * a)


if __name__ == '__main__':
    t = np.linspace(-5, 5, 6)
    t1, t2 = np.meshgrid(t, t)
    x1 = np.stack((t1.ravel(), t2.ravel()), axis=1)
    N = len(x1)
    x2 = x1 + (1, 1)
    x = np.concatenate((x1, x2))
    y = np.array([1] * N + [-1] * N)

    clf = svm.SVC(C=0.1, kernel='rbf', gamma=5)
    clf.fit(x, y)
    y_hat = clf.predict(x)
    print('Accuracy:%.3f%%' % (np.mean(y_hat == y) * 100))

    # 开始画图
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b'])
    x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max())
    x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max())
    # 生成网格采样点
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    y_test = clf.predict(x_test)
    y_test = y_test.reshape(x1.shape)

    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_test, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=cm_dark, alpha=0.7)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)
    plt.tight_layout(pad=2.5)
    plt.title('SVM Metholds RBF Overfiting', fontsize=18)
    plt.show()