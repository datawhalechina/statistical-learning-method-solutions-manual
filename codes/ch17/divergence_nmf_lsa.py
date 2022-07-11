#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: divergence_nmf_lsa.py
@time: 2022/7/11 16:50
@project: statistical-learning-method-solutions-manual
@desc: 习题17.2 损失函数是散度损失时的非负矩阵分解算法
"""
import numpy as np


class DivergenceNmfLsa:
    def __init__(self, max_iter=1000, tol=1e-6, random_state=0):
        """
        损失函数是散度损失时的非负矩阵分解
        :param max_iter: 最大迭代次数
        :param tol: 容差
        :param random_state: 随机种子
        """
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _init_param(self, X, k):
        self.__m, self.__n = X.shape
        self.__W = np.random.random((self.__m, k))
        self.__H = np.random.random((k, self.__n))

    def _div_loss(self, X, W, H):
        Y = np.dot(W, H)
        loss = 0
        for i in range(self.__m):
            for j in range(self.__n):
                loss += (X[i][j] * np.log(X[i][j] / Y[i][j]) if X[i][j] * Y[i][j] > 0 else 0) - X[i][j] + Y[i][j]

        return loss

    def fit(self, X, k):
        """
        :param X: 单词-文本矩阵
        :param k: 话题个数
        :return:
        """
        # (1)初始化
        self._init_param(X, k)
        # (2.c)计算散度损失
        loss = self._div_loss(X, self.__W, self.__H)

        for _ in range(self.max_iter):
            # (2.a)更新W的元素
            WH = np.dot(self.__W, self.__H)
            for i in range(self.__m):
                for l in range(k):
                    s1 = sum(self.__H[l][j] * X[i][j] / WH[i][j] for j in range(self.__n))
                    s2 = sum(self.__H[l][j] for j in range(self.__n))
                    self.__W[i][l] *= s1 / s2

            # (2.b)更新H的元素
            WH = np.dot(self.__W, self.__H)
            for l in range(k):
                for j in range(self.__n):
                    s1 = sum(self.__W[i][l] * X[i][j] / WH[i][j] for i in range(self.__m))
                    s2 = sum(self.__W[i][l] for i in range(self.__m))
                    self.__H[l][j] *= s1 / s2

            new_loss = self._div_loss(X, self.__W, self.__H)
            if abs(new_loss - loss) < self.tol:
                break

            loss = new_loss

        return self.__W, self.__H


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
    k = 3
    div_nmf = DivergenceNmfLsa(max_iter=1000, random_state=2022)
    W, H = div_nmf.fit(X, k)
    print("话题空间W：")
    print(W)
    print("文本在话题空间的表示H：")
    print(H)
