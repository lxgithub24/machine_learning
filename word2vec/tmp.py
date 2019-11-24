# -*- coding: utf-8 -*-
# @Time    : 2019/1/24 16:52
# @Author  : RIO
# @desc: TODO:DESC
# https://www.cnblogs.com/pinard/p/7278324.html
import time, datetime

interface_list = list()
with open('./tmp.log', 'r') as f:
    lines = f.readlines()
    for line in lines:
        interface_list.append(line.strip().replace('/cangjie_gateway/', ''))
print(list(set(interface_list)))