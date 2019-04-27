# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 10:54
# @Author  : RIO
# @desc: TODO:DESC
a = [1, 2, 2]*8


def permutation(a, start):
    if start == len(a) - 1:
        # print(a)
        pass
    else:
        tmp_set = set()
        for i in range(start, len(a)):
            if a[i] in tmp_set:
                continue
            tmp_set.add(a[i])
            _swap(a, start, i)
            permutation(a, start + 1)
            _swap(a, start, i)


def _swap(a, i, j):
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp


def main1():
    import time
    st = time.time()
    permutation(a, 0)
    print(time.time() - st)
    return 1
main1()