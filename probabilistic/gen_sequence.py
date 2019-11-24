# -*- coding: utf-8 -*-
# @Time    : 2019/11/22 11:12
# @Author  : RIO
# @desc: TODO:DESC
import random, copy

r_total_num = 10
r_n = 3
seq_num = 5
permutation_m_from_seq_list = []
tmp_permutation_m_from_seq_list = []
adjacent_seq_num = 3
current_seq_id = 10
history_seq = []


def gen_r_seq():
    return random.sample([i for i in range(1, 1 + r_total_num)], r_n)


def merge_exclude():
    tmp_list = []
    for i in range(seq_num):
        tmp_list += gen_r_seq()
    return [i for i in range(1, 1 + r_total_num) if i not in tmp_list and i not in remove_current_seq(current_seq_id)]


def remove_current_seq(i):
    if i < adjacent_seq_num:
        return
    tmp_list = []
    for idx in range(i - adjacent_seq_num, i):
        tmp_list += history_seq[idx]
    return list(set(tmp_list))


def permutation_m_from_seq(seq, num):
    if len(tmp_permutation_m_from_seq_list) == r_n:
        permutation_m_from_seq_list.append(copy.copy(tmp_permutation_m_from_seq_list))
        return
    for i in range(len(seq)):
        tmp_permutation_m_from_seq_list.append(seq[i])
        permutation_m_from_seq(seq[i + 1:], num - 1)
        tmp_permutation_m_from_seq_list.pop()


if __name__ == '__main__':
    ret = gen_r_seq()
    print(ret)
