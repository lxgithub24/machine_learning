# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 11:41
# @Author  : RIO
# @desc: 逻辑回归调包
from sklearn.linear_model import LogisticRegressionCV
from numpy import array

# X, y = [[1, 0], [2, 0], [2, 1], [100, 2], [0, 1], [1, 2], [1, 3], [3, 200]], [0, 0, 0, 0, 1, 1, 1, 1]
# lr_clf = LogisticRegressionCV(Cs=10, random_state=0).fit(X, y)
# print(lr_clf.coef_)
# print(lr_clf.predict_proba([[2, 3]]))
# print(lr_clf.predict([[2, 3]]))




X, y = [[1, 0], [2, 0], [2, 1], [100, 2], [0, 1], [1, 2], [1, 3], [3, 200]], [0, 0, 0, 0, 1, 1, 1, 1]
lr_clf = LogisticRegressionCV(Cs=array([1.00000000e-02, 1.62377674e-02, 2.63665090e-02, 4.28133240e-02,
       6.95192796e-02, 1.12883789e-01, 1.83298071e-01, 2.97635144e-01,
       4.83293024e-01, 7.84759970e-01, 1.27427499e+00, 2.06913808e+00,
       3.35981829e+00, 5.45559478e+00, 8.85866790e+00, 1.43844989e+01,
       2.33572147e+01, 3.79269019e+01, 6.15848211e+01, 1.00000000e+02]),
                     class_weight='balanced', cv=5, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=100, multi_class='ovr', n_jobs=None, penalty='l2',
                     random_state=None, refit=True, scoring=None,
                     solver='lbfgs', tol=0.01, verbose=0).fit(X, y)
print(lr_clf.coef_)
print(lr_clf.predict_proba([[2, 3]]))
print(lr_clf.predict([[2, 3]]))