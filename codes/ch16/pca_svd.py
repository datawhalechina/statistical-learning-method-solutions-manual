#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: pca_svd.py
@time: 2022/7/14 10:48
@project: statistical-learning-method-solutions-manual
@desc: 习题16.1 样本矩阵的奇异值分解的主成分分析算法
"""

import numpy as np


def pca_svd(X, k):
    """
    样本矩阵的奇异值分解的主成分分析算法
    :param X: 样本矩阵X
    :param k: 主成分个数k
    :return: 特征向量V，样本主成分矩阵Y
    """
    n_samples = X.shape[1]

    # 构造新的n×m矩阵
    T = X.T / np.sqrt(n_samples - 1)

    # 对矩阵T进行截断奇异值分解
    U, S, V = np.linalg.svd(T)
    V = V[:, :k]

    # 求k×n的样本主成分矩阵
    return V, np.dot(V.T, X)


if __name__ == '__main__':
    X = np.array([[2, 3, 3, 4, 5, 7],
                  [2, 4, 5, 5, 6, 8]])
    X = X.astype("float64")

    # 规范化变量
    avg = np.average(X, axis=1)
    var = np.var(X, axis=1)
    for i in range(X.shape[0]):
        X[i] = (X[i, :] - avg[i]) / np.sqrt(var[i])

    # 设置精度为3
    np.set_printoptions(precision=3, suppress=True)
    V, vnk = pca_svd(X, 2)

    print("正交矩阵V：")
    print(V)
    print("样本主成分矩阵Y：")
    print(vnk)
