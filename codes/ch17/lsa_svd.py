#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lsa_svd.py
@time: 2022/7/11 15:21
@project: statistical-learning-method-solutions-manual
@desc: 习题17.1 根据题意，利用矩阵奇异值分解进行潜在语义分析
"""

import numpy as np


def lsa_svd(X, k):
    """
    潜在语义分析的矩阵奇异值分解
    :param X: 单词-文本矩阵
    :param k: 话题数
    :return: 话题向量空间、文本集合在话题向量空间的表示
    """
    # 单词-文本矩阵X的奇异值分解
    U, S, V = np.linalg.svd(X)
    # 矩阵的截断奇异值分解，取前k个
    U = U[:, :k]
    S = np.diag(S[:k])
    V = V[:k, :]

    return U, np.dot(S, V)


if __name__ == '__main__':
    X = np.array([[2, 0, 0, 0],
                  [0, 2, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 2, 3],
                  [0, 0, 0, 1],
                  [1, 2, 2, 1]])

    # 设置精度为2
    np.set_printoptions(precision=2, suppress=True)
    # 假设话题的个数是3个
    U, SV = lsa_svd(X, k=3)
    print("话题空间U：")
    print(U)
    print("文本在话题空间的表示SV：")
    print(SV)
