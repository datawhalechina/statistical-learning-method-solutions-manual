#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: zero_sum_game.py
@time: 2023/3/17 18:48
@project: statistical-learning-method-solutions-manual
@desc: 习题28.2 零和博弈的代码验证
"""

import numpy as np


def minmax_function(A):
    """
    从收益矩阵中计算minmax的算法
    :param A: 收益矩阵
    :return: 计算得到的minmax结果
    """
    index_max = []
    for i in range(len(A)):
        # 计算每一行的最大值
        index_max.append(A[i, :].max())

    # 计算每一行的最大值中的最小值
    minmax = min(index_max)
    return minmax


def maxmin_function(A):
    """
    从收益矩阵中计算maxmin的算法
    :param A: 收益矩阵
    :return: 计算得到的maxmin结果
    """
    column_min = []
    for i in range(len(A)):
        # 计算每一列的最小值
        column_min.append(A[:, i].min())

    # 计算每一列的最小值中的最大值
    maxmin = max(column_min)
    return maxmin


if __name__ == '__main__':
    # 创建收益矩阵
    A = np.array([[-1, 2], [4, 1]])
    # 计算maxmin
    maxmin = maxmin_function(A)
    # 计算minmax
    minmax = minmax_function(A)
    # 输出结果
    print("maxmin =", maxmin)
    print("minmax =", minmax)
