#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: svd.py
@time: 2022/7/15 10:06
@project: statistical-learning-method-solutions-manual
@desc: 习题15.1 自编程实现奇异值分解
"""
import numpy as np
from scipy.linalg import null_space


def my_svd(A):
    m = A.shape[0]

    # (1) 计算对称矩阵 A^T A 的特征值与特征向量，
    W = np.dot(A.T, A)
    # 返回的特征值lambda_value是升序的，特征向量V是单位化的特征向量
    lambda_value, V = np.linalg.eigh(W)
    # 并按特征值从大到小排列
    lambda_value = lambda_value[::-1]
    lambda_value = lambda_value[lambda_value > 0]
    # (2) 计算n阶正价矩阵V
    V = V[:, -1::-1]

    # (3) 求 m * n 对角矩阵
    sigma = np.sqrt(lambda_value)
    S = np.diag(sigma) @ np.eye(*A.shape)

    # (4.1) 求A的前r个正奇异值
    r = np.linalg.matrix_rank(A)
    U1 = np.hstack([(np.dot(A, V[:, i]) / sigma[i])[:, np.newaxis] for i in range(r)])
    # (4.2) 求A^T的零空间的一组标准正交基
    U = U1
    if r < m:
        U2 = null_space(A.T)
        U2 = U2[:, r:]
        U = np.hstack([U, U2])

    return U, S, V


if __name__ == '__main__':
    A = np.array([[1, 2, 0],
                  [2, 0, 2]])

    np.set_printoptions(precision=2, suppress=True)

    U, S, V = my_svd(A)
    print("U=", U)
    print("S=", S)
    print("V=", V)
    calc = np.dot(np.dot(U, S), V.T)
    print("A=", calc)
