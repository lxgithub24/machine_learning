# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 21:15
# @Author  : RIO
# @desc: TODO:DESC

with open('./tmp', 'r') as fp:
    for data in fp:
        split = data.split()
        print(split)