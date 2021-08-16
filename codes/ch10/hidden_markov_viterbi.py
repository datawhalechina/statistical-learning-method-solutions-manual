#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: hidden_markov_viterbi.py
@time: 2021/8/16 23:13
@project: statistical-learning-method-solutions-manual
@desc: 习题10.3 隐马尔可夫模型的维特比算法
"""

import numpy as np


class HiddenMarkovViterbi:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def viterbi(self, Q, V, A, B, O, PI):
        """
        维特比算法
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
        # 初始化deltas
        deltas = np.zeros((N, M))
        # 初始化psis
        psis = np.zeros((N, M))

        # 初始化最优路径矩阵，该矩阵维度与观测序列维度相同
        I = np.zeros((1, M))
        # (2)递推，遍历观测序列
        for t in range(M):
            if self.verbose:
                if t == 0:
                    print("初始化Psi1和delta1：")
                elif t == 1:
                    print("\n从时刻2到T的所有单个路径中概率最大值delta和概率最大的路径的第t-1个结点Psi：")

            # (2)递推从t=2开始
            real_t = t + 1
            # 得到序列对应的索引
            index_of_o = V.index(O[t])
            for i in range(N):
                real_i = i + 1
                if t == 0:
                    # (1)初始化
                    deltas[i][t] = PI[0][i] * B[i][index_of_o]
                    psis[i][t] = 0

                    self.print_delta_t1(B, PI, deltas, i, index_of_o, real_i, t)
                    self.print_psi_t1(real_i)
                else:
                    # (2)递推，对t=2,3,...,T
                    deltas[i][t] = np.max(np.multiply([delta[t - 1] for delta in deltas],
                                                      [a[i] for a in A])) * B[i][index_of_o]
                    self.print_delta_t(A, B, deltas, i, index_of_o, real_i, real_t, t)

                    psis[i][t] = np.argmax(np.multiply([delta[t - 1] for delta in deltas],
                                                       [a[i] for a in A]))
                    self.print_psi_t(i, psis, real_i, real_t, t)

        last_deltas = [delta[M - 1] for delta in deltas]
        # (3)终止，得到所有路径的终结点最大的概率值
        P = np.max(last_deltas)
        # (3)得到最优路径的终结点
        I[0][M - 1] = np.argmax(last_deltas)
        if self.verbose:
            print("\n所有路径的终结点最大的概率值：")
            print("P = %f" % P)
        if self.verbose:
            print("\n最优路径的终结点：")
            print("i%d = argmax[deltaT(i)] = %d" % (M, I[0][M - 1] + 1))
            print("\n最优路径的其他结点：")

        # (4)递归由后向前得到其他结点
        for t in range(M - 2, -1, -1):
            I[0][t] = psis[int(I[0][t + 1])][t + 1]
            if self.verbose:
                print("i%d = Psi%d(i%d) = %d" % (t + 1, t + 2, t + 2, I[0][t] + 1))

        # 输出最优路径
        print("\n最优路径是：", "->".join([str(int(i + 1)) for i in I[0]]))

    def print_psi_t(self, i, psis, real_i, real_t, t):
        if self.verbose:
            print("Psi%d(%d) = argmax[delta%d(j) * aj%d] = %d"
                  % (real_t, real_i, real_t - 1, real_i, psis[i][t]))

    def print_delta_t(self, A, B, deltas, i, index_of_o, real_i, real_t, t):
        if self.verbose:
            print("delta%d(%d) = max[delta%d(j) * aj%d] * b%d(o%d) = %.2f * %.2f = %.5f"
                  % (real_t, real_i, real_t - 1, real_i, real_i, real_t,
                     np.max(np.multiply([delta[t - 1] for delta in deltas],
                                        [a[i] for a in A])),
                     B[i][index_of_o], deltas[i][t]))

    def print_psi_t1(self, real_i):
        if self.verbose:
            print("Psi1(%d) = 0" % real_i)

    def print_delta_t1(self, B, PI, deltas, i, index_of_o, real_i, t):
        if self.verbose:
            print("delta1(%d) = pi%d * b%d(o1) = %.2f * %.2f = %.2f"
                  % (real_i, real_i, real_i, PI[0][i], B[i][index_of_o], deltas[i][t]))


if __name__ == '__main__':
    Q = [1, 2, 3]
    V = ['红', '白']
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    O = ['红', '白', '红', '白']
    PI = [[0.2, 0.4, 0.4]]

    HMM = HiddenMarkovViterbi(verbose=True)
    HMM.viterbi(Q, V, A, B, O, PI)
