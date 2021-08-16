#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: hidden_markov_backward.py
@time: 2021/8/16 21:17
@project: statistical-learning-method-solutions-manual
@desc: 习题10.1 隐马尔可夫模型的后向算法
"""

import numpy as np


class HiddenMarkovBackward:
    def __init__(self, verbose=False):
        self.betas = None
        self.backward_P = None
        self.verbose = verbose

    def backward(self, Q, V, A, B, O, PI):
        """
        后向算法
        :param Q: 所有可能的状态集合
        :param V: 所有可能的观测集合
        :param A: 状态转移概率矩阵
        :param B: 观测概率矩阵
        :param O: 观测序列
        :param PI: 初始状态概率向量
        """
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # (1)初始化后向概率beta值，书中第201页公式(10.19)
        betas = np.ones((N, M))
        if self.verbose:
            self.print_betas_T(N, M)

        # (2)对观测序列逆向遍历，M-2即为T-1
        if self.verbose:
            print("\n从时刻T-1到1观测序列的后向概率：")
        for t in range(M - 2, -1, -1):
            # 得到序列对应的索引
            index_of_o = V.index(O[t + 1])
            # 遍历状态序列
            for i in range(N):
                # 书中第201页公式(10.20)
                betas[i][t] = np.dot(np.multiply(A[i], [b[index_of_o] for b in B]),
                                     [beta[t + 1] for beta in betas])
                real_t = t + 1
                real_i = i + 1
                if self.verbose:
                    self.print_betas_t(A, B, N, betas, i, index_of_o, real_i, real_t, t)

        # 取出第一个值索引，用于得到o1
        index_of_o = V.index(O[0])
        self.betas = betas
        # 书中第201页公式(10.21)
        P = np.dot(np.multiply(PI, [b[index_of_o] for b in B]),
                   [beta[0] for beta in betas])
        self.backward_P = P
        self.print_P(B, N, P, PI, betas, index_of_o)

    @staticmethod
    def print_P(B, N, P, PI, betas, index_of_o):
        print("\n观测序列概率：")
        print("P(O|lambda) = ", end="")
        for i in range(N):
            print("%.1f * %.1f * %.5f + "
                  % (PI[0][i], B[i][index_of_o], betas[i][0]), end="")
        print("0 = %f" % P)

    @staticmethod
    def print_betas_t(A, B, N, betas, i, index_of_o, real_i, real_t, t):
        print("beta%d(%d) = sum[a%dj * bj(o%d) * beta%d(j)] = ("
              % (real_t, real_i, real_i, real_t + 1, real_t + 1), end='')
        for j in range(N):
            print("%.2f * %.2f * %.2f + "
                  % (A[i][j], B[j][index_of_o], betas[j][t + 1]), end='')
        print("0) = %.3f" % betas[i][t])

    @staticmethod
    def print_betas_T(N, M):
        print("初始化后向概率：")
        for i in range(N):
            print('beta%d(%d) = 1' % (M, i + 1))


if __name__ == '__main__':
    Q = [1, 2, 3]
    V = ['红', '白']
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    O = ['红', '白', '红', '白']
    PI = [[0.2, 0.4, 0.4]]

    hmm_backward = HiddenMarkovBackward()
    hmm_backward.backward(Q, V, A, B, O, PI)
