# -*- coding: utf-8 -*-
# @Time    : 2019/7/20 21:12
# @Author  : RIO
# @desc: 负采样中：正负样本的训练过程
from numpy import zeros, random,exp, dot, outer
def train_sentence_sg(model, sentence, alpha, work=None):
    """
    对于每个单词列表(sentence)进行训练
    syn0 是隐层(某个词的词向量), 维度为layer1_size
    fb 是输出层, 维度为k + 1
    syn1neg 是隐层(某个词的词向量)到输出层的权重, 维度为layer1_size * (k+1)
    与一般的神经网络不同的是, 这里的隐层和隐层到输出层的权重都是参数, 都需要优化
    """
    if model.negative:
        # labels存储的是正样本(当前word)和model.negative(k)个负样本的label, word的label为1, 负样本的label为0
        labels = zeros(model.negative + 1)
        labels[0] = 1.0

    for pos, word in enumerate(sentence):
        if word is None:
            continue
            # 窗口的大小是[0, window]之间的一个随机数
        reduced_window = random.randint(model.window)

        # 根据word附近rediced_window窗口内的某个word2, 预测word本身以及k个负样本
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start: pos + model.window + 1 - reduced_window], start):
            if word2 and not (pos2 == pos):
                l1 = model.syn0[word2.index]
                neu1e = zeros(l1.shape)

                if model.negative:
                    # 构建要预测的word和k个负样本的词向量
                    word_indices = [word.index]
                    while len(word_indices) < model.negative + 1:
                        w = model.table[random.randint(model.table.shape[0])]
                        if w != word.index:
                            word_indices.append(w)
                    # 隐层(word2的词向量, 有layer1_size个节点)到输出层(k+1个节点)的权重, 是一个(k+1)*layer1_size的矩阵
                    l2b = model.syn1neg[word_indices]
                    # label与输出的误差 * 学习率
                    fb = 1. / (1. + exp(-dot(l1, l2b.T)))
                    # 隐层向输出层的前向反馈
                    gb = (labels - fb) * alpha
                    # 更新权重
                    model.syn1neg[word_indices] += outer(gb, l1)
                    # 计算隐层的梯度(隐层代表的是word2的词向量, 也是参数, 所以也需要更新)
                    neu1e += dot(gb, l2b)  # save error
                # 更新隐层, 亦即word2的词向量
                model.syn0[word2.index] += neu1e

    return len([word for word in sentence if word is not None])