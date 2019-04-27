# -*- coding: utf-8 -*-
# @Time    : 2019/4/2 20:52
# @Author  : RIO
# @desc: 跳台阶问题
N =10
dp = [0]*(N+1)
dp[1] = 1
dp[2] = 2
for i in range(3, N+1):
    dp[i] = dp[i-1] + dp[i-2]
print(dp)