#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: em_plsa.py
@time: 2022/7/13 18:48
@project: statistical-learning-method-solutions-manual
@desc: 习题18.3 基于生成模型的EM算法的概率潜在语义分析
"""

import numpy as np


class EMPlsa:
    def __init__(self, max_iter=100, random_state=2022):
        """
        基于生成模型的EM算法的概率潜在语义模型
        :param max_iter: 最大迭代次数
        :param random_state: 随机种子
        """
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, K):
        """
        :param X: 单词-文本矩阵
        :param K: 话题个数
        :return: P(w_i|z_k) 和 P(z_k|d_j)
        """
        # M, N分别为单词个数和文本个数
        M, N = X.shape

        # 计算n(d_j)
        n_d = [np.sum(X[:, j]) for j in range(N)]

        # (1)设置参数P(w_i|z_k)和P(z_k|d_j)的初始值
        np.random.seed(self.random_state)
        p_wz = np.random.random((M, K))
        p_zd = np.random.random((K, N))

        # (2)迭代执行E步和M步，直至收敛为止
        for _ in range(self.max_iter):
            # E步
            P = np.zeros((M, N, K))
            for i in range(M):
                for j in range(N):
                    for k in range(K):
                        P[i][j][k] = p_wz[i][k] * p_zd[k][j]
                    P[i][j] /= np.sum(P[i][j])

            # M步
            for k in range(K):
                for i in range(M):
                    p_wz[i][k] = np.sum([X[i][j] * P[i][j][k] for j in range(N)])
                p_wz[:, k] /= np.sum(p_wz[:, k])

            for k in range(K):
                for j in range(N):
                    p_zd[k][j] = np.sum([X[i][j] * P[i][j][k] for i in range(M)]) / n_d[j]

        return p_wz, p_zd


if __name__ == "__main__":
    # 输入文本-单词矩阵，共有9个文本，11个单词
    X = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 2, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0]])

    # 设置精度为3
    np.set_printoptions(precision=3, suppress=True)

    # 假设话题的个数是3个
    k = 3

    em_plsa = EMPlsa(max_iter=100)

    p_wz, p_zd = em_plsa.fit(X, 3)

    print("参数P(w_i|z_k)：")
    print(p_wz)
    print("参数P(z_k|d_j)：")
    print(p_zd)
