# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 11:05
# @Author  : RIO
# @desc: TODO:DESC
import torch.nn as nn
import torch
hidden2tag = nn.Linear(4, 5)
nn_parameter = nn.Parameter(torch.randn(5, 5))
print(nn_parameter)
nn_parameter.data[3, :] = -10000
nn_parameter.data[:, 4] = -10000
print(nn_parameter)

