# -*- coding: utf-8 -*-
# @Time    : 2019/1/24 17:11
# @Author  : RIO
# @desc: TODO:DESC

from gensim.models import Word2Vec
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)