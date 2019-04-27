# -*- coding: utf-8 -*-
# @Time    : 2019/2/16 11:28
# @Author  : RIO
# @desc: TODO:DESC

# http://www.tsinsen.com/A1228###
#
# 4 4
# 0 0 0 0
# 1 2 2 0
# 0 2 2 1
# 0 0 0 0
# 5 5 5 5
# 5 5 5 5
# 5 5 5 5
# 5 5 5 5
# 2 1 3 4 2 2
#
# Z
# 15
#
# 20% N, M ≤ 10; Bij ≤ 20
# 40% N, M ≤ 100; Bij ≤ 20
# 100% 1 ≤ N, M ≤ 150; 0 ≤ Bij ≤ 109; 0 ≤ Aij ≤ 1000


def get_min_cost(n, m, aij, bij, x, y, z):

    pass


def dp(aij, bij):

    pass


if __name__ == '__main__':
    n = input()
    m = input()

    aij = []
    for i in n:
        row = []
        for j in m:
            row.append(input())
        aij.append(row)

    bij = []
    for i in n:
        row = []
        for j in m:
            row.append(input())
        bij.append(row)

    x = (input(), input())
    y = (input(), input())
    z = (input(), input())

    get_min_cost(n, m, aij, bij, x, y, z)