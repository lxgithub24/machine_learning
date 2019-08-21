# -*- coding: utf-8 -*-
# @Time    : 2019/8/13 16:16
# @Author  : RIO
# @desc: TODO:DESC
import numpy as np
from sklearn import svm
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt


def show_accuracy(a, b):
    acc = a.ravel() == b.ravel()
    print('the acc:%.3f%%' % (100 * float(acc.sum()) / a.size))


if __name__ == '__main__':
    data = np.loadtxt('bipartition.txt', dtype=np.float, delimiter='\t')
    x, y = np.split(data, (2,), axis=1)
    y[y == 0] = -1
    y = y.ravel()
    print(x)
    print(y)

    # 设置分类器,进行调参
    clf_param = (('linear', 0.1), ('linear', 0.5), ('linear', 1), ('linear', 2),
                 ('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
                 ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100))

    # 开始画图
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    # 生成网格采样点
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    # 测试点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b'])

    plt.figure(figsize=(14, 14), facecolor='w')
    for i, param in enumerate(clf_param):
        # print(param)
        clf = svm.SVC(C=param[1], kernel=param[0])
        if param[0] == 'rbf':
            clf.gamma = param[2]
            title = 'Ga K, C=%.1f, gamma=%.1f' % (param[1], param[2])
        else:
            title = 'linear kernel, C=%.1f' % param[1]

        clf.fit(x, y)
        y_hat = clf.predict(x)
        show_accuracy(y_hat, y)

        # 画图
        print(title)
        print('the numbers of support vectors:\n', clf.n_support_)
        print('the params of support vectors:\n', clf.dual_coef_)
        print('the support vectors:\n', clf.support_)
        plt.subplot(3, 4, i + 1)
        grid_hat = clf.predict(grid_test)
        grid_hat = grid_hat.reshape(x1.shape)
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)
        plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k', s=40, cmap=cm_dark)
        plt.scatter(x[clf.support_, 0], x[clf.support_, 1], edgecolor='k', facecolor='none', s=100, marker='o')
        z = clf.decision_function(grid_test)
        z = z.reshape(x1.shape)
        plt.contour(x1, x2, z, colors=list('kmrmk'), linestyles=['--', '-', '--', '-', '--'],
                    lw=[1, 0.5, 1.5, 0.5, 1], levels=[-1, -0.5, 0, 0.5, 1])
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(title, fontsize=13)
    plt.suptitle('different params in svm classification', fontsize=20)
    plt.tight_layout(2.0)
    plt.subplots_adjust(top=0.90)
    plt.show()