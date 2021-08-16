#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: hidden_markov_forward_backward.py
@time: 2021/8/16 22:08
@project: statistical-learning-method-solutions-manual
@desc: 习题10.2 隐马尔可夫模型的前向后向算法
"""

import numpy as np
from hidden_markov_backward import HiddenMarkovBackward


class HiddenMarkovForwardBackward(HiddenMarkovBackward):
    def __init__(self, verbose=False):
        super(HiddenMarkovBackward, self).__init__()
        self.alphas = None
        self.forward_P = None
        self.verbose = verbose

    def forward(self, Q, V, A, B, O, PI):
        """
        前向算法
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
        # 初始化前向概率alpha值
        alphas = np.zeros((N, M))
        # 时刻数=观测序列数
        T = M
        # (2)对观测序列遍历，遍历每一个时刻，计算前向概率alpha值

        for t in range(T):
            if self.verbose:
                if t == 0:
                    print("前向概率初值：")
                elif t == 1:
                    print("\n从时刻1到T-1观测序列的前向概率：")
            # 得到序列对应的索引
            index_of_o = V.index(O[t])
            # 遍历状态序列
            for i in range(N):
                if t == 0:
                    # (1)初始化alpha初值，书中第198页公式(10.15)
                    alphas[i][t] = PI[t][i] * B[i][index_of_o]
                    if self.verbose:
                        self.print_alpha_t1(alphas, i, t)
                else:
                    # (2)递推，书中第198页公式(10.16)
                    alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas],
                                          [a[i] for a in A]) * B[i][index_of_o]
                    if self.verbose:
                        self.print_alpha_t(alphas, i, t)
        # (3)终止，书中第198页公式(10.17)
        self.forward_P = np.sum([alpha[M - 1] for alpha in alphas])
        self.alphas = alphas

    @staticmethod
    def print_alpha_t(alphas, i, t):
        print("alpha%d(%d) = [sum alpha%d(i) * ai%d] * b%d(o%d) = %f"
              % (t + 1, i + 1, t - 1, i, i, t, alphas[i][t]))

    @staticmethod
    def print_alpha_t1(alphas, i, t):
        print('alpha1(%d) = pi%d * b%d * b(o1) = %f'
              % (i + 1, i, i, alphas[i][t]))

    def calc_t_qi_prob(self, t, qi):
        result = (self.alphas[qi - 1][t - 1] * self.betas[qi - 1][t - 1]) / self.backward_P[0]
        if self.verbose:
            print("计算P(i%d=q%d|O,lambda)：" % (t, qi))
            print("P(i%d=q%d|O,lambda) = alpha%d(%d) * beta%d(%d) / P(O|lambda) = %f"
                  % (t, qi, t, qi, t, qi, result))

        return result


if __name__ == '__main__':
    Q = [1, 2, 3]
    V = ['红', '白']
    A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    O = ['红', '白', '红', '红', '白', '红', '白', '白']
    PI = [[0.2, 0.3, 0.5]]

    hmm_forward_backward = HiddenMarkovForwardBackward(verbose=True)
    hmm_forward_backward.forward(Q, V, A, B, O, PI)
    print()
    hmm_forward_backward.backward(Q, V, A, B, O, PI)
    print()
    hmm_forward_backward.calc_t_qi_prob(t=4, qi=3)
