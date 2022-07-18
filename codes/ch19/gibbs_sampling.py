#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: gibbs_sampling_demo.py
@time: 2022/7/18 17:22
@project: statistical-learning-method-solutions-manual
@desc: 习题19.8 使用吉布斯抽样算法估计参数的均值和方差
"""
import matplotlib.pyplot as plt
import numpy as np


class GibbsSampling:
    def __init__(self, target_dist, j, m=1e4, n=1e5):
        """
        Gibbs Sampling 算法

        :param target_dist: 目标分布
        :param j: 变量维度
        :param m: 收敛步数
        :param n: 迭代步数
        """
        self.target_dist = target_dist
        self.j = j
        self.m = int(m)
        self.n = int(n)

    def solve(self):
        """
        Gibbs Sampling 算法求解
        """
        # (1) 初始化
        all_samples = np.zeros((self.n, self.j))
        # 任意选择一个初始值
        x_0 = np.random.random(self.j)
        # x_0 = np.array([0.5, 0.5])
        # (2) 循环执行
        for i in range(self.n):
            x = x_0 if i == 0 else all_samples[i - 1]
            # 满条件分布抽取
            for k in range(self.j):
                x[k] = self.target_dist.sample(x, k)
            all_samples[i] = x
        # (3) 得到样本集合
        samples = all_samples[self.m:]
        # (4) 计算函数样本均值
        dist_mean = samples.mean(0)
        dist_var = samples.var(0)
        return samples[self.m:], dist_mean, dist_var

    @staticmethod
    def visualize(samples, bins=50):
        """
        可视化展示
        :param samples: 抽取的随机样本集合
        :param bins: 频率直方图的分组个数
        """
        fig, ax = plt.subplots()
        ax.set_title('Gibbs Sampling')
        ax.hist(samples[:, 0], bins, alpha=0.7, label='$\\theta$')
        ax.hist(samples[:, 1], bins, alpha=0.7, label='$\\eta$')
        ax.set_xlim(0, 1)
        ax.legend()
        plt.show()


class TargetDistribution:
    """
    目标概率分布
    """

    def sample(self, x, k=0):
        """
        从满条件分布中抽取新的分量 x_j
        """
        theta, eta = x
        if k == 0:
            while True:
                new_theta = np.random.uniform(0, 1 - eta)
                alpha = np.random.uniform()
                if alpha < self.__prob([new_theta, eta]):
                    return new_theta
        elif k == 1:
            while True:
                new_eta = np.random.uniform(0, 1 - theta)
                alpha = np.random.uniform()
                if alpha < self.__prob([theta, new_eta]):
                    return new_eta

    @staticmethod
    def __prob(x):
        """
        P(X = x) 的概率
        """
        theta = x[0]
        eta = x[1]
        p1 = (theta / 4 + 1 / 8) ** 14
        p2 = theta / 4
        p3 = eta / 4
        p4 = (eta / 4 + 3 / 8)
        p5 = 1 / 2 * (1 - theta - eta) ** 5
        p = (p1 * p2 * p3 * p4 * p5)
        # 联合概率值过小，可对其进行放缩
        return p * 1e12


if __name__ == '__main__':
    # 收敛步数
    m = 1e2
    # 迭代步数
    n = 1e3

    # 目标分布
    target_dist = TargetDistribution()

    # 使用 Gibbs Sampling 算法进行求解
    gibbs_sampling = GibbsSampling(target_dist, 2, m, n)

    samples, dist_mean, dist_var = gibbs_sampling.solve()

    print(f'theta均值：{dist_mean[0]}, theta方差：{dist_var[0]}')
    print(f'eta均值：{dist_mean[1]}, eta方差：{dist_var[1]}')

    # 对结果进行可视化
    GibbsSampling.visualize(samples, bins=20)
