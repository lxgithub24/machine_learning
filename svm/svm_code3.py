# -*- coding: utf-8 -*-
# @Time    : 2019/8/13 16:16
# @Author  : RIO
# @desc: TODO:DESC
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 50
    np.random.seed(0)
    x = np.sort(np.random.uniform(0, 6, N), axis=0)
    y = 2 * np.sin(x) + 0.1 * np.random.randn(N)
    x = x.reshape(-1, 1)
    print('x:\n', x)
    print('y:\n', y)

    print('--------SVR RBF---------')
    svr_rbf_model = svm.SVR(kernel='rbf', gamma=0.2, C=100)
    svr_rbf_model.fit(x, y)

    print('--------SVR Linear---------')
    svr_linear_model = svm.SVR(kernel='linear', C=100)
    svr_linear_model.fit(x, y)

    print('--------SVR Polynomial---------')
    svr_poly_model = svm.SVR(kernel='poly', degree=3, C=100)
    svr_poly_model.fit(x, y)

    x_test = np.linspace(x.min(), 1.5 * x.max(), 100).reshape(-1, 1)
    y_hat_rbf = svr_rbf_model.predict(x_test)
    y_hat_linear = svr_linear_model.predict(x_test)
    y_hat_poly = svr_poly_model.predict(x_test)

    plt.figure(figsize=(10, 10), facecolor='w')
    # plt.scatter(x[sp], y[sp], s=120, c='r', marker='*', label='Support Vectors', zorder=3)
    plt.plot(x_test, y_hat_rbf, 'r-', lw=2, label='RBF Kernel')
    plt.plot(x_test, y_hat_linear, 'g-', lw=2, label='linear Kernel')
    plt.plot(x_test, y_hat_poly, 'b-', lw=2, label='poly Kernel')
    plt.plot(x, y, 'go', markersize=5)
    plt.legend(loc='upper right')
    plt.title('1.1 SVR', fontsize=16)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()