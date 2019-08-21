# encoding: utf-8
import random

# 前序 池子大小
total_num1 = 100
# 后序 池子大小
total_num2 = 30
# 前序序列长度
sequence1_len = 50
# 后序序列长度
sequence2_len = 15
# 迭代次数
_iter = 5


def get_res():
    # 获取序列
    def get_list():
        L = [random.randint(1, total_num1) for _ in range(sequence1_len)]
        return L


    # get_abandon_list():
    s = set()
    for i in range(_iter):
        _l = get_list()
        s = s.union(_l)


    # get_selected_list():
    _l = [i for i in range(1, total_num1 + 1) if i not in s]


    # get_res_list():
    res = [_l[random.randint(1, len(_l))] for _ in range(sequence1_len)]
    return res
print(get_res())