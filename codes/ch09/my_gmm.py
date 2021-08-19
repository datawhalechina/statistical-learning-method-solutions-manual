#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: my_gmm.py
@time: 2021/8/14 2:46
@project: statistical-learning-method-solutions-manual
@desc: 习题9.3 自编程实现求两个分量的高斯混合模型的5个参数
"""

import numpy as np
import itertools


class MyGMM:
    def __init__(self, alphas_init, means_init, covariances_init, tol=1e-6, n_components=2, max_iter=50):
        # (1)设置参数的初始值
        # 分模型权重
        self.alpha_ = np.array(alphas_init, dtype="float16").reshape(n_components, 1)
        # 分模型均值
        self.mean_ = np.array(means_init, dtype="float16").reshape(n_components, 1)
        # 分模型标准差（方差的平方）
        self.covariances_ = np.array(covariances_init, dtype="float16").reshape(n_components, 1)
        # 迭代停止的阈值
        self.tol = tol
        # 高斯混合模型分量个数
        self.K = n_components
        # 最大迭代次数
        self.max_iter = max_iter
        # 观测数据
        self._y = None
        # 实际迭代次数
        self.n_iter_ = 0

    def gaussian(self, mean, convariances):
        """计算高斯分布概率密度"""
        return 1 / np.sqrt(2 * np.pi * convariances) * np.exp(
            -(self._y - mean) ** 2 / (2 * convariances))

    def update_r(self, mean, convariances, alpha):
        """更新r_jk, 分模型k对观测数据yi的响应度"""
        r_jk = alpha * self.gaussian(mean, convariances)
        return r_jk / r_jk.sum(axis=0)

    def update_params(self, r):
        """更新mean, alpha, covariances每个分模型k的均值、权重、方差的平方"""
        u = self.mean_[-1]
        _mean = ((r * self._y).sum(axis=1) / r.sum(axis=1)).reshape(self.K, 1)
        _covariances = ((r * (self._y - u) ** 2).sum(axis=1) / r.sum(axis=1)).reshape(self.K, 1)
        _alpha = (r.sum(axis=1) / self._y.size).reshape(self.K, 1)
        return _mean, _covariances, _alpha

    def judge_stop(self, mean, covariances, alpha):
        """中止条件判断"""
        a = np.linalg.norm(self.mean_ - mean)
        b = np.linalg.norm(self.covariances_ - covariances)
        c = np.linalg.norm(self.alpha_ - alpha)
        return True if np.sqrt(a ** 2 + b ** 2 + c ** 2) < self.tol else False

    def fit(self, y):
        self._y = np.copy(np.array(y))
        """迭代训练获得预估参数"""
        # (2)E步：计算分模型k对观测数据yi的响应度
        # 更新r_jk, 分模型k对观测数据yi的响应度
        r = self.update_r(self.mean_, self.covariances_, self.alpha_)
        # 更新mean, alpha, covariances每个分模型k的均值、权重、方差的平方
        _mean, _covariances, _alpha = self.update_params(r)
        for i in range(self.max_iter):
            if not self.judge_stop(_mean, _covariances, _alpha):
                # (4)未达到阈值条件，重复迭代
                r = self.update_r(_mean, _covariances, _alpha)
                # (3)M步：计算新一轮迭代的模型参数
                _mean, _covariances, _alpha = self.update_params(r)
            else:
                # 达到阈值条件，停止迭代
                self.n_iter_ = i
                break

            self.mean_ = _mean
            self.covariances_ = _covariances
            self.alpha_ = _alpha

    def score(self):
        """计算该局部最优解的score，即似然函数值"""
        return (self.alpha_ * self.gaussian(self.mean_, self.covariances_)).sum()


if __name__ == "__main__":
    # 观测数据
    y = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]).reshape(1, 15)
    # 预估均值和方差，以其邻域划分寻优范围
    y_mean = y.mean() // 1
    y_std = (y.std() ** 2) // 1

    # 网格搜索，对不同的初值进行参数估计
    alpha = [[i, 1 - i] for i in np.linspace(0.1, 0.9, 9)]
    mean = [[y_mean + i, y_mean + j] for i in range(-10, 10, 5) for j in range(-10, 10, 5)]
    covariances = [[y_std + i, y_std + j] for i in range(-1000, 1000, 500) for j in range(-1000, 1000, 500)]
    results = []
    for i in itertools.product(alpha, mean, covariances):
        init_alpha = i[0]
        init_mean = i[1]
        init_covariances = i[2]
        clf = MyGMM(alphas_init=init_alpha, means_init=init_mean, covariances_init=init_covariances,
                    n_components=2, tol=1e-6)
        clf.fit(y)
        # 得到不同初值收敛的局部最优解
        results.append([clf.alpha_, clf.mean_, clf.covariances_, clf.score()])
    # 根据score，从所有局部最优解找到相对最优解
    best_value = max(results, key=lambda x: x[3])

    print("alpha : {}".format(best_value[0].T))
    print("mean : {}".format(best_value[1].T))
    print("std : {}".format(best_value[2].T))
