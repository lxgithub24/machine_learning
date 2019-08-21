# -*- coding: utf-8 -*-
# @Time    : 2019/8/11 18:50
# @Author  : RIO
# @desc: https://blog.csdn.net/qq_36523839/article/details/82347332

# --------------------------------------------svd demo------------------------------------------------------------------
# SCV实现的相关线性代数，但我们无需担心SVD的实现，在Numpy中有一个称为线性代数linalg的线性代数工具箱能帮助我们。下面演示其用法对于一个简单的矩阵：
from numpy import *
from numpy import linalg as la

df = mat(array([[1, 1], [1, 7]]))
U, Sigma, VT = la.svd(df)
print(U)
# [[ 0.16018224  0.98708746]
#  [ 0.98708746 -0.16018224]]
print(Sigma)
# [7.16227766 0.83772234]
print(VT)
# [[ 0.16018224  0.98708746]
#  [ 0.98708746 -0.16018224]]
# --------------------------------------------svd demo------------------------------------------------------------------

# --------------------------------------------svd demo------------------------------------------------------------------
# 上面代码种用了三种计算距离的函数，经过测试后使用其中一种便可以了。然后对于物品评分函数中的nonzero(logical_and)不是很明白的请看这篇专门讲解的文章。以上为普通的处理方式，下面使用SVD来做基于物品协同过滤。
from numpy import *
from numpy import linalg as la


# （用户x商品）    # 为0表示该用户未评价此商品，即可以作为推荐商品
def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 0, 5, 0, 0],
            [1, 1, 1, 0, 0]]


# !!!假定导入数据都为列向量，若行向量则需要对代码简单修改

# 欧几里德距离 这里返回结果已处理 0，1   0最大相似，1最小相似   欧氏距离转换为2范数计算
def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


# 皮尔逊相关系数 numpy的corrcoef函数计算
def pearsSim(inA, inB):
    if (len(inA) < 3):
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]  # 使用0.5+0.5*x 将-1，1 转为 0，1


# 余玄相似度 根据公式带入即可，其中分母为2范数计算，linalg的norm可计算范数
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)  # 同样操作转换 0，1


# 对物品评分  (数据集 用户行号 计算误差函数 推荐商品列号)
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]  # 获得特征列数
    simTotal = 0.0;
    ratSimTotal = 0.0  # 两个计算估计评分值变量初始化
    for j in range(n):
        userRating = dataMat[user, j]  # 获得此人对该物品的评分
        if userRating == 0:  # 若此人未评价过该商品则不做下面处理
            continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]  # 获得相比较的两列同时都不为0的数据行号
        if len(overLap) == 0:
            similarity = 0
        else:
            # 求两列的相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])  # 利用上面求得的两列同时不为0的行的列向量 计算距离
        # print('%d 和 %d 的相似度是: %f' % (item, j, similarity))
        simTotal += similarity  # 计算总的相似度
        ratSimTotal += similarity * userRating  # 不仅仅使用相似度，而是将评分当权值*相似度 = 贡献度
    if simTotal == 0:  # 若该推荐物品与所有列都未比较则评分为0
        return 0
    else:
        return ratSimTotal / simTotal  # 归一化评分 使其处于0-5（评级）之间


# 给出推荐商品评分
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # 找到该行所有为0的位置（即此用户未评价的商品，才做推荐）
    if len(unratedItems) == 0:
        return '所有物品都已评价...'
    itemScores = []
    for item in unratedItems:  # 循环所有没有评价的商品列下标
        estimatedScore = estMethod(dataMat, user, simMeas, item)  # 计算当前产品的评分
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]  # 将推荐商品排序


# 结果测试如下：
myMat = mat(loadExData())
myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4  # 将数据某些值替换，增加效果
myMat[3, 3] = 2
result1 = recommend(myMat, 2)  # 余玄相似度
print(result1)
result2 = recommend(myMat, 2, simMeas=ecludSim)  # 欧氏距离
print(result2)
result3 = recommend(myMat, 2, simMeas=pearsSim)  # 皮尔逊相关度
print(result3)
