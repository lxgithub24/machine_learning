# -*- coding: utf-8 -*-
# @Time    : 2019/11/22 11:39
# @Author  : RIO
# @desc: TODO:DESC
import copy
r_n = 3

permutation_m_from_seq_list = []
tmp_permutation_m_from_seq_list = []


def permutation_m_from_seq(seq, num):
    if len(tmp_permutation_m_from_seq_list) == r_n:
        permutation_m_from_seq_list.append(copy.copy(tmp_permutation_m_from_seq_list))
        return
    for i in range(len(seq)):
        tmp_permutation_m_from_seq_list.append(seq[i])
        permutation_m_from_seq(seq[i + 1:], num - 1)
        tmp_permutation_m_from_seq_list.pop()


permutation_m_from_seq([1, 2, 3, 4, 5], 3)
print(permutation_m_from_seq_list)
