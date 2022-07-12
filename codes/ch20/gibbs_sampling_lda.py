#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: gibbs_sampling_lda.py
@time: 2022/7/12 20:30
@project: statistical-learning-method-solutions-manual
@desc: 习题20.2 LDA吉布斯抽样算法
"""

import numpy as np


class GibbsSamplingLDA:
    def __init__(self, iter_max=1000):
        self.iter_max = iter_max
        self.weights_ = []

    def fit(self, words, K):
        """
        :param words: 单词-文本矩阵
        :param K: 话题个数
        :return: 文本话题序列z
        """
        # M, Nm分别为文本个数和单词个数
        words = words.T
        M, Nm = words.shape

        # 初始化超参数alpha, beta，其中alpha为文本的话题分布相关参数，beta为话题的单词分布相关参数
        alpha = np.array([1 / K] * K)
        beta = np.array([1 / Nm] * Nm)

        # 初始化参数theta, varphi，其中theta为文本关于话题的多项分布参数，varphi为话题关于单词的多项分布参数
        theta = np.zeros([M, K])
        varphi = np.zeros([K, Nm])

        # 输出文本的话题序列z
        z = np.zeros(words.shape, dtype='int')

        # (1)设所有计数矩阵的元素n_mk、n_kv，计数向量的元素n_m、n_k初值为 0
        n_mk = np.zeros([M, K])
        n_kv = np.zeros([K, Nm])
        n_m = np.zeros(M)
        n_k = np.zeros(K)

        # (2)对所有M个文本中的所有单词进行循环
        for m in range(M):
            for v in range(Nm):
                # 如果单词v存在于文本m
                if words[m, v] != 0:
                    # (2.a)抽样话题
                    z[m, v] = np.random.choice(list(range(K)))
                    # 增加文本-话题计数
                    n_mk[m, z[m, v]] += 1
                    # 增加文本-话题和计数
                    n_m[m] += 1
                    # 增加话题-单词计数
                    n_kv[z[m, v], v] += 1
                    # 增加话题-单词和计数
                    n_k[z[m, v]] += 1

        # (3)对所有M个文本中的所有单词进行循环，直到进入燃烧期
        zi = 0
        for i in range(self.iter_max):
            for m in range(M):
                for v in range(Nm):
                    # (3.a)如果单词v存在于文本m，那么当前单词是第v个单词，话题指派z_mv是第k个话题
                    if words[m, v] != 0:
                        # 减少计数
                        n_mk[m, z[m, v]] -= 1
                        n_m[m] -= 1
                        n_kv[z[m, v], v] -= 1
                        n_k[z[m, v]] -= 1

                        # (3.b)按照满条件分布进行抽样
                        max_zi_value, max_zi_index = -float('inf'), z[m, v]
                        for k in range(K):
                            zi = ((n_kv[k, v] + beta[v]) / (n_kv[k, :].sum() + beta.sum())) * \
                                 ((n_mk[m, k] + alpha[k]) / (n_mk[m, :].sum() + alpha.sum()))

                        # 得到新的第 k‘个话题，分配给 z_mv
                        if max_zi_value < zi:
                            max_zi_value, max_zi_index = zi, k
                            z[m, v] = max_zi_index

                        # (3.c) (3.d)增加计数并得到两个更新的计数矩阵的n_kv和n_mk
                        n_mk[m, z[m, v]] += 1
                        n_m[m] += 1
                        n_kv[z[m, v], v] += 1
                        n_k[z[m, v]] += 1

        # (4)利用得到的样本计数，计算模型参数
        for m in range(M):
            for k in range(K):
                theta[m, k] = (n_mk[m, k] + alpha[k]) / (n_mk[m, :].sum() + alpha.sum())

        for k in range(K):
            for v in range(Nm):
                varphi[k, v] = (n_kv[k, v] + beta[v]) / (n_kv[k, :].sum() + beta.sum())

        self.weights_ = [varphi, theta]
        return z.T, n_kv, n_mk


if __name__ == '__main__':
    gibbs_sampling_lda = GibbsSamplingLDA(iter_max=1000)

    # 输入文本-单词矩阵，共有9个文本，11个单词
    words = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0],
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

    K = 3  # 假设话题数量为3

    # 设置精度为3
    np.set_printoptions(precision=3, suppress=True)

    z, n_kv, n_mk = gibbs_sampling_lda.fit(words, K)
    varphi = gibbs_sampling_lda.weights_[0]
    theta = gibbs_sampling_lda.weights_[1]

    print("文本的话题序列z：")
    print(z)
    print("样本的计数矩阵N_KV：")
    print(n_kv)
    print("样本的计数矩阵N_MK：")
    print(n_mk)
    print("模型参数varphi：")
    print(varphi)
    print("模型参数theta：")
    print(theta)
