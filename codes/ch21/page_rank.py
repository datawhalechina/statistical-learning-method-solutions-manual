#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: page_rank.py
@time: 2022/7/18 9:14
@project: statistical-learning-method-solutions-manual
@desc: 习题21.2 基本定义的PageRank的迭代算法
"""

import numpy as np


def page_rank_basic(M, R0, max_iter=1000):
    """
    迭代求解基本定义的PageRank
    :param M: 转移矩阵
    :param R0: 初始分布向量
    :param max_iter: 最大迭代次数
    :return: Rt: 极限向量
    """
    Rt = R0
    for _ in range(max_iter):
        Rt = np.dot(M, Rt)
    return Rt


if __name__ == '__main__':
    # 使用例21.1的转移矩阵M
    M = np.array([[0, 1 / 2, 1, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    # 使用5个不同的初始分布向量R0
    for _ in range(5):
        R0 = np.random.rand(4)
        R0 = R0 / np.linalg.norm(R0, ord=1)
        Rt = page_rank_basic(M, R0)
        print("R0 =", R0)
        print("Rt =", Rt)
        print()