# -*- coding: utf-8 -*-
# @Time    : 2019/1/24 16:52
# @Author  : RIO
# @desc: TODO:DESC
# https://www.cnblogs.com/pinard/p/7278324.html
import jieba.analyse
from gensim.models import word2vec
import os
import pickle

# 当前工作地址
curr_dir = os.path.dirname(__file__)


# 结巴配置
def jieba_config():
    jieba.suggest_freq('沙瑞金', True)
    jieba.suggest_freq('田国富', True)
    jieba.suggest_freq('高育良', True)
    jieba.suggest_freq('侯亮平', True)
    jieba.suggest_freq('钟小艾', True)
    jieba.suggest_freq('陈岩石', True)
    jieba.suggest_freq('欧阳菁', True)
    jieba.suggest_freq('易学习', True)
    jieba.suggest_freq('王大路', True)
    jieba.suggest_freq('蔡成功', True)
    jieba.suggest_freq('孙连城', True)
    jieba.suggest_freq('季昌明', True)
    jieba.suggest_freq('丁义珍', True)
    jieba.suggest_freq('郑西坡', True)
    jieba.suggest_freq('赵东来', True)
    jieba.suggest_freq('高小琴', True)
    jieba.suggest_freq('赵瑞龙', True)
    jieba.suggest_freq('林华华', True)
    jieba.suggest_freq('陆亦可', True)
    jieba.suggest_freq('刘新建', True)
    jieba.suggest_freq('刘庆祝', True)


# 处理数据，获取w2v需要的数据
def data_process():
    jieba_config()
    sa = set()
    with open(curr_dir + './data/in_the_name_of_people.txt', encoding='utf-8') as f:
        document = f.read()
        # document_decode = document.decode('GBK')
        document_cut = jieba.cut(document)
        # print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
        result = ' '.join(document_cut)
        for i in result.split(' '):
            sa.add(i)
        with open(curr_dir + './data/in_the_name_of_people_segment.txt', 'w', encoding='utf-8') as f2:
            f2.write(result)


# 训练模型
def train():
    sentences = word2vec.LineSentence('./data/in_the_name_of_people_segment.txt')
    model = word2vec.Word2Vec(sentences, hs=1, negative=0, min_count=1, window=3, size=100, sg=1)
    with open(curr_dir + './data/w2v.model', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model():
    with open(curr_dir + './data/w2v.model', "rb") as f:
        model = pickle.load(f, encoding='utf-8')
    return model


# 第一个是最常用的，找出某一个词向量最相近的词集合，
def get_topn_of_word():
    model = load_model()
    req_count = 5
    for key in model.wv.similar_by_word('沙瑞金', topn=100):
        if len(key[0]) == 3:
            req_count -= 1
            print(key[0], key[1])
            if req_count == 0:
                break


# 计算词向量的均值等
def word_embed_operation():
    model = load_model()
    # sharuijin = model.wv['沙瑞金']
    # gaoyuliang = model.wv['高育良']
    print(model.wv.syn0[0])
    # print('-' * 90)
    # print(sharuijin)
    # print('-' * 90)
    # print(gaoyuliang)
    # print('-' * 90)
    # print(type(model.wv['沙瑞金']))
    # print(sharuijin + gaoyuliang)


def get_word_similarity():
    model = load_model()
    # 第二个应用是看两个词向量的相近程度，这里给出了书中两组人的相似程度：
    print(model.wv.similarity('沙瑞金', '高育良'))
    print(model.wv.similarity('李达康', '王大路'))
    # 第三个应用是找出不同类的词，这里给出了人物分类题：
    print(model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split()))
    print('=' * 35)
    print(model.accuracy())


if __name__ == '__main__':
    # train()
    word_embed_operation()
