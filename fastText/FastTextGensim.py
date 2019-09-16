# -*- coding: utf-8 -*-
# @Time    : 2019/9/14 16:56
# @Author  : RIO
# @desc: TODO:DESC
# 参考地址：https://radimrehurek.com/gensim/models/fasttext.html
# from gensim.test.utils import common_texts  # some example sentences
# from gensim.models import FastText
#
# print(common_texts[0])
#
# print(len(common_texts))
#
# model = FastText(size=4, window=3, min_count=1)  # instantiate
# model.build_vocab(sentences=common_texts)
# model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)  # train




import fasttext

# Skipgram model
model = fasttext.train_unsupervised("train.txt", model='skipgram', lr=0.05, dim=100, ws=5, epoch=5)
model.save_model("model_file.bin")
