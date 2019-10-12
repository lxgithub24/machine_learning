# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 9:05
# @Author  : RIO
# @desc: 参考：https://www.cnblogs.com/jyroy/p/9427977.html
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()
# f = open("iris.data.csv", 'wb')              #可以保存数据
# f.write(str(iris))
# f.close()
print(iris)
knn.fit(iris.data, iris.target)  # 用KNN的分类器进行建模，这里利用的默认的参数，大家可以自行查阅文档
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print("predictedLabel is :" + predictedLabel)
