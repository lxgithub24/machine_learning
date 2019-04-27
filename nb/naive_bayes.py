from numpy import *
from math import *
from pylab import *


# load example data
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


# create a list of all the unique words in all of our documents
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 文档的向量(inputSet)是否在词汇表(vacabList)中,则将输出向量的下标变为1
# 将文字转化为关于0,1的向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 朴素贝叶斯分类器训练函数1
def trainNB0(trainMatrix, trainCategory):
    # print("trainMatrix:%s\ntrainCategory:%s"%(trainMatrix,trainCategory))
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # print("numTrainDocs=%d\tnumWords=%d"%(numTrainDocs,numWords))
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 3/6
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)  # 长度为32的0向量
    # print("pAbusive=%d\np0Num:%s\np1Num:%s"%(pAbusive,p0Num,p1Num))
    p0Denom = 0.0
    p1Denom = 0.0  # 所有文档中，属于类别0和1的词汇个数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            # print("p1Num:",p1Num)
            p1Denom += sum(trainMatrix[i])
            # print("p1Denom",p1Denom)
        else:
            p0Num += trainMatrix[i]
            # print("p0Num:",p0Num)
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom  # change to log()
    p0Vect = p0Num / p0Denom  # change to log()
    # print("p1Vect:",p1Vect)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类器训练函数--改进版
def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 文档数
    numWords = len(trainMatrix[0])  # 词汇表的词汇数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 带侮辱性质类别的文档数
    p0Num = ones(numWords)  # 是侮辱性质类别的向量
    p1Num = ones(numWords)  # 长度为32的0向量
    p0Denom = 2.0
    p1Denom = 2.0  # 所有文档中，属于类别0和1的词汇个数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 类别1的词频
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]  # 类别0的词频
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 分类器测试
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


testingNB()
