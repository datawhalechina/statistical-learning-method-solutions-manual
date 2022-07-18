#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: metropolis_hastings_demo.py
@time: 2022/7/18 16:05
@project: statistical-learning-method-solutions-manual
@desc: 习题19.7 使用Metropolis-Hastings算法求后验概率分布的均值和方差
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binom


class MetropolisHastings:
    def __init__(self, proposal_dist, accepted_dist, m=1e4, n=1e5):
        """
        Metropolis Hastings 算法

        :param proposal_dist: 建议分布
        :param accepted_dist: 接受分布
        :param m: 收敛步数
        :param n: 迭代步数
        """
        self.proposal_dist = proposal_dist
        self.accepted_dist = accepted_dist
        self.m = m
        self.n = n

    @staticmethod
    def __calc_acceptance_ratio(q, p, x, x_prime):
        """
        计算接受概率

        :param q: 建议分布
        :param p: 接受分布
        :param x: 上一状态
        :param x_prime: 候选状态
        """
        prob_1 = p.prob(x_prime) * q.joint_prob(x_prime, x)
        prob_2 = p.prob(x) * q.joint_prob(x, x_prime)
        alpha = np.min((1., prob_1 / prob_2))
        return alpha

    def solve(self):
        """
        Metropolis Hastings 算法求解
        """
        all_samples = np.zeros(self.n)
        # (1) 任意选择一个初始值
        x_0 = np.random.random()
        # (2) 循环执行
        for i in range(int(self.n)):
            x = x_0 if i == 0 else all_samples[i - 1]
            # (2.a) 从建议分布中抽样选取
            x_prime = self.proposal_dist.sample()
            # (2.b) 计算接受概率
            alpha = self.__calc_acceptance_ratio(self.proposal_dist, self.accepted_dist, x, x_prime)
            # (2.c) 从区间 (0,1) 中按均匀分布随机抽取一个数 u
            u = np.random.uniform(0, 1)
            # 根据 u <= alpha，选择 x 或 x_prime 进行赋值
            if u <= alpha:
                all_samples[i] = x_prime
            else:
                all_samples[i] = x

        # (3) 随机样本集合
        samples = all_samples[self.m:]
        # 函数样本均值
        dist_mean = samples.mean()
        # 函数样本方差
        dist_var = samples.var()
        return samples[self.m:], dist_mean, dist_var

    @staticmethod
    def visualize(samples, bins=50):
        """
        可视化展示
        :param samples: 抽取的随机样本集合
        :param bins: 频率直方图的分组个数
        """
        fig, ax = plt.subplots()
        ax.set_title('Metropolis Hastings')
        ax.hist(samples, bins, alpha=0.7, label='Samples Distribution')
        ax.set_xlim(0, 1)
        ax.legend()
        plt.show()


class ProposalDistribution:
    """
    建议分布
    """

    @staticmethod
    def sample():
        """
        从建议分布中抽取一个样本
        """
        # B(1,1)
        return beta.rvs(1, 1, size=1)

    @staticmethod
    def prob(x):
        """
        P(X = x) 的概率
        """
        return beta.pdf(x, 1, 1)

    def joint_prob(self, x_1, x_2):
        """
        P(X = x_1, Y = x_2) 的联合概率
        """
        return self.prob(x_1) * self.prob(x_2)


class AcceptedDistribution:
    """
    接受分布
    """

    @staticmethod
    def prob(x):
        """
        P(X = x) 的概率
        """
        # Bin(4, 10)
        return binom.pmf(4, 10, x)


if __name__ == '__main__':
    # 收敛步数
    m = 1000
    # 迭代步数
    n = 10000

    # 建议分布
    proposal_dist = ProposalDistribution()
    # 接受分布
    accepted_dist = AcceptedDistribution()

    metropolis_hastings = MetropolisHastings(proposal_dist, accepted_dist, m, n)

    # 使用 Metropolis-Hastings 算法进行求解
    samples, dist_mean, dist_var = metropolis_hastings.solve()
    print("均值:", dist_mean)
    print("方差:", dist_var)

    # 对结果进行可视化
    metropolis_hastings.visualize(samples, bins=20)