# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 10:54
# @Author  : RIO
# @desc: TODO:DESC

a = [8, 7, 3, 2, 6, 1, 5, 9, 4, 10]
# 总共多少个数字
total = 10
# 选出多少个数字
_len = 4
# 最接近的和
sum = 18


def quick_sort(a, start, end):
    left = start
    right = end
    curr_axe = end
    axe_value = a[end]
    while left < right:
        # 右移直到遇到大于axe,交换
        while left < curr_axe:
            if a[left] <= axe_value:
                left += 1
            else:
                a[curr_axe] = a[left]
                curr_axe = left
                break
        # 左移直到遇到小于axe
        while right > curr_axe:
            if a[right] >= axe_value:
                right -= 1
            else:
                a[curr_axe] = a[right]
                curr_axe = right
                break
        a[curr_axe] = axe_value
    # 如果左侧长度大于1，左侧快排
    if start < curr_axe - 1:
        quick_sort(a, start, curr_axe - 1)
    # 如果右侧长度大于1，右侧快拍
    if end > curr_axe + 1:
        quick_sort(a, curr_axe + 1, end)


quick_sort(a, 0, len(a) - 1)
print(a)
# 当前序列和
tmp_sum = 0

for j in range(_len):
    tmp_sum += a[j]
# print('test_cangjie_gateway_interface', tmp_sum)
# 当前序列和和目标值差距
min_value = abs(sum - tmp_sum)
# 和最接近的初始值
min_value_start = 0
for i in range(1, total - _len + 1):
    tmp_sum = tmp_sum - a[i - 1] + a[i + 3]
    if abs(tmp_sum - sum) < min_value:
        min_value = abs(tmp_sum - sum)
        min_value_start = i
# print(min_value_start)
# print(min_value)
