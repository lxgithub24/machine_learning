# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 11:11
# @Author  : RIO
# @desc: TODO:DESC

from __future__ import print_function
from glove import Glove
from glove import Corpus

# 准备数据集
sentense = [['你', '是', '谁'], ['我', '是', '中国人']]
corpus_model = Corpus()
corpus_model.fit(sentense, window=10)
# corpus_model.save('corpus.model')
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)

# 训练
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

# glove模型保存与加载：
glove.save('glove.model')
glove = Glove.load('glove.model')
corpus_model.save('corpus.model')
corpus_model = Corpus.load('corpus.model')

# 根据glove求相似词。
glove.most_similar('我', number=10)

# 词向量矩阵
# 全部词向量矩阵
print(glove.word_vectors)
# 指定词条词向量
print(glove.word_vectors[glove.dictionary['你']])
# 语料协同矩阵 corpus coocurrence matrix
corpus_model.matrix.todense().tolist()
