# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 16:45
# @Author  : RIO
# @desc: TODO:DESC
import numpy as np
import time
from sklearn import datasets
iris = datasets.load_iris()
#print(iris.data)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#预测自己的数据
start1 = time.clock()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
end1 = time.clock()
print('运行的时间是:',end1-start1)
print(iris.data.shape[0],(iris.target != y_pred).sum())
'''这个预测的公式比较简单，就是计算这个公式的值。
连续属性：拟合一个高斯模型，属性值代入高斯模型得到概率，对于每个c有个高斯函数，这个是要求出来的
上面这种方法的话，不好对应，但是如果每个都算一遍重复有太多。连续值暂时不考虑了。连续值还是可以的，只要求出
那两个参数即可。因此，主要是求高斯的拟合，公式书上有。
input:医生的数据
output:
'''

'''input:2darray
   output:[[某属性，类别1的系数：a,b],[],[],[]]
   某个类别中的某个属性的值集合，也就是分表操作。这次是竖着分表.直接上numpy
   高斯分布的系数直接用公式
'''
def getGaussianCoeff(data_X,data_y):
    values= list(set(data_y.tolist()))  #需要分表的个数
    #print(values)
    box =[]
    coeff = []
    data = np.column_stack((data_X,data_y)) #X，y连接在一起
  #  print(data)
    for value in values:   #按照值进行分表
        zhongjie = []
        for i in range(len(data)):
            if data[i][-1]==value:
                zhongjie.append(data[i].tolist())
        box.append(zhongjie)
    #对每个属性，计算高斯分布的系数
    box = np.array(box) #分表转成ndarray，这是多维数组(3,50,5)
    for i in range(len(values)):#对于每个分表
        sigema =[]
        uVector = np.average(box[i],axis=0) #均值向量,axis表示竖着求均值
        #print(uVector)
        uVecyorMinusAvg = np.array([(box[i][j]-uVector) for j in range(len(box[i]))])#x-u
        uVecyorMinusAvgTranspose = uVecyorMinusAvg.T
        xieFangChaMatrix = np.dot(uVecyorMinusAvg,uVecyorMinusAvgTranspose)#算出协方差
        #遍历出[1,1],[2,2]...这些位置上的数字，也就是各个sigema
        for j in range(len(xieFangChaMatrix)):
            for k in range(len(xieFangChaMatrix)):
                if j ==k:
                  sigema.append(round(xieFangChaMatrix[j][k],3))
        coeff.append(list(zip(sigema,uVector)))
    return coeff
'''input: preData : ndarray or list
   return list
   0,先计算出P1的概率，就是出现某种结果的概率
   1，预测过程就是取出每个数据
   2，对于可能的每个种类计算出相关概率，概率最大的就是那个类
      2.1 相关概率=此情况概率P1*每个属性的高斯分布概率P2
      2.2取出最大的概率即可
'''
def gaussiFunc(sigema,u,x):#这里的sigema是平方的形式
    outxishu = 1/(np.sqrt(2*np.pi*sigema))
    inxishu = -((x-u)**2)/(2*sigema)
    return outxishu*np.exp(inxishu)

def trainBayes(Data_X,Data_Y,preData):
    #求出高斯函数的系数
    coeff = getGaussianCoeff(Data_X,Data_Y)
    if len(Data_X)!=len(Data_Y):
        raise TypeError
    if not isinstance(Data_X,list):#转成list
        Data_X = Data_X.tolist()
        Data_Y = Data_Y.tolist()
    numOfPreData = len(preData)  #需要预测的个数
    values = list(set(Data_Y)) #分表的个数
    preDataY=[]
    #计算出数据归为这个类的概率，
    counts = [0]*len(values)
    for k in range(len(Data_X)):
        for value in values:
            if Data_Y[k]==value:
                counts[values.index(value)]=counts[values.index(value)]+1
    P1s =[counts[l]/sum(counts) for l in range(len(values))] #计算出P1的概率
    for i in range(numOfPreData):
        probabilities = []
        for j in range(len(values)):  #每种结果的概率
            #用上面的高斯分布求每个概率，然后相乘
            P2 = 1
            for m in range(len(Data_X[0])):
                myCoeff = coeff[j][m]  #是一个tuple,第一个元素是sigema，第二个参数是平均值u
                probaOfOneAtrr = gaussiFunc(myCoeff[0],myCoeff[1],preData[i][m])#第i个数据，第m个属性的高斯概率值
                P2 = P2*probaOfOneAtrr
            probabilities.append(P2*P1s[j])#把所有可能的结果的概率放入
        probabilities = np.array(probabilities)
        index = np.argmax(probabilities) #选取那个最大的概率
        preDataY.append(index)  #每个数据的结果写入
    return preDataY
'''
x = np.linspace(-10,10,100)#高斯函数检测
y = gaussiFunc(1,2,x)
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()           
'''
start2 = time.clock()
preData = trainBayes(iris.data,iris.target,iris.data)
print(preData)
end2 = time.clock()
print('我的程序运行的时间是：',end2 -start2)
print((iris.target != preData).sum())
