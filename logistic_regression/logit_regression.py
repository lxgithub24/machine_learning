# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 11:41
# @Author  : RIO
# @desc: 逻辑回归调包
from sklearn.linear_model import LogisticRegressionCV


X, y = [[1, 0], [2, 0], [2, 1], [100, 2], [0, 1], [1, 2], [1, 3], [3, 200]], [0, 0, 0, 0, 1, 1, 1, 1]
lr_clf = LogisticRegressionCV(Cs=10, random_state=0).fit(X, y)
print(lr_clf.coef_)
print(lr_clf.predict_proba([[2, 3]]))
print(lr_clf.predict([[2, 3]]))


# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegressionCV
# X, y = load_iris(return_X_y=True)
# clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X, y)
# print(clf.predict(X[:2, :]))

