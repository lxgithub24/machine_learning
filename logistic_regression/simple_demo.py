# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 11:11
# @Author  : RIO
# @desc: 按照逻辑回归的原理，完全手动实现逻辑回归

from numpy import exp, array, random, dot, sum, max


# 逻辑回归模型
class LogisticRegression():
    # 初始化
    def __init__(self):
        # 确定随机数种子，每次产生的随机数都一样，方便debug
        random.seed(1)
        # 生成随机数作为初始权重值
        self.synaptic_weights = 2 * random.random((2, 1)) - 1
        # 学习率
        self.learning_rate = 0.0001

    # 定义激活函数sigmoid
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # 训练
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # 得到模型输出
            output = self.predict(training_set_inputs)
            # 计算误差
            error = output - training_set_outputs
            # 调整权重值: θ = θ - alpha * (1/m)*sum((h(xi)-yi)*xi)
            adjustment = dot(training_set_inputs.T, error)
            self.synaptic_weights -= self.learning_rate * sum(adjustment) / len(training_set_inputs)

    # 模型 y=sigmoid(w1*x1+w2*x2)
    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    logistic_regression = LogisticRegression()

    # 训练集合
    training_set_inputs = array([[1, 0], [2, 0], [2, 1], [100, 2], [0, 1], [1, 2], [1, 3], [3, 200]])
    training_set_outputs = array([[0, 0, 0, 0, 1, 1, 1, 1]]).T

    # 特征缩放：min-max标准化
    feature0_max = max(training_set_inputs.all(), 0)
    feature0_min = min(training_set_inputs.all(), 0)
    training_set_inputs_zoom = (training_set_inputs - feature0_min) / (feature0_max - feature0_min)
    print("feature zoom max size: ", feature0_max)
    print("feature zoom min size: ", feature0_min)
    print("training set inputs with zoom: \n", training_set_inputs_zoom)

    # 训练10000次
    logistic_regression.train(training_set_inputs_zoom, training_set_outputs, 10000)

    # 测试神经网络
    print("Considering new situation [2,3] -> ?: ")
    print(logistic_regression.predict(array([2, 3])))
