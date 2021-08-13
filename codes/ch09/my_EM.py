#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: my_EM.py
@time: 2021/8/14 2:46
@project: statistical-learning-method-solutions-manual
@desc: 习题9.3 求两个分量的高斯混合模型的5个参数
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import time


class MyEM:
    def __init__(self, y, args, tol=0.001, K=2, max_iter=50):
        self.y = y
        self.r = None  # 分模型k对观测数据yi的响应度
        self.al = [np.array(args[0], dtype="float16").reshape(K, 1)]  # 分模型权重
        self.u = [np.array(args[1], dtype="float16").reshape(K, 1)]  # 分模型均值
        self.si = [np.array(args[2], dtype="float16").reshape(K, 1)]  # 分模型方差的平方
        self.score = 0  # 局部最优解评分
        self.tol = tol  # 迭代中止阈值
        self.K = K  # 高斯混合模型分量个数
        self.max_iter = max_iter  # 最大迭代次数

    def gaussian(self):
        """计算高斯分布概率密度"""
        return 1 / np.sqrt(2 * np.pi * self.si[-1]) * np.exp(-(self.y - self.u[-1]) ** 2 / (2 * self.si[-1]))

    def updata_r(self):
        """更新r_jk 分模型k对观测数据yi的响应度"""
        r_jk = self.al[-1] * self.gaussian()
        self.r = r_jk / r_jk.sum(axis=0)

    def updata_u_al_si(self):
        """更新u al si 每个分模型k的均值、权重、方差的平方"""
        u = self.u[-1]
        self.u.append(((self.r * self.y).sum(axis=1) / self.r.sum(axis=1)).reshape(self.K, 1))
        self.si.append((((self.r * (self.y - u) ** 2).sum(axis=1) / self.r.sum(axis=1))).reshape(self.K, 1))
        self.al.append((self.r.sum(axis=1) / self.y.size).reshape(self.K, 1))

    def judge_stop(self):
        """中止条件判断"""
        a = np.linalg.norm(self.u[-1] - self.u[-2])
        b = np.linalg.norm(self.si[-1] - self.si[-2])
        c = np.linalg.norm(self.al[-1] - self.al[-2])
        return True if np.sqrt(a ** 2 + b ** 2 + c ** 2) < self.tol else False

    def fit(self):
        """迭代训练获得预估参数"""
        self.updata_r()  # 更新r_jk 分模型k对观测数据yi的响应度
        self.updata_u_al_si()  # 更新u al si 每个分模型k的均值、权重、方差的平方
        for i in range(self.max_iter):
            if not self.judge_stop():  # 若未达到中止条件，重复迭代
                self.updata_r()
                self.updata_u_al_si()
            else:  # 若达到中止条件，计算该局部最优解的“评分”，迭代中止
                self._score()
                break

    def _score(self):
        """计算该局部最优解的'评分'，也就是似然函数值"""
        self.score = (self.al[-1] * self.gaussian()).sum()


if __name__ == "__main__":
    star = time.time()
    y = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]).reshape(1, 15)
    # 预估均值和方差，以其邻域划分寻优范围
    y_mean = y.mean() // 1
    y_std = (y.std() ** 2) // 1
    al = [[i, 1 - i] for i in np.linspace(0.1, 0.9, 9)]
    u = [[y_mean + i, y_mean + j] for i in range(-10, 10, 5) for j in range(-10, 10, 5)]
    si = [[y_std + i, y_std + j] for i in range(-1000, 1000, 500) for j in range(-1000, 1000, 500)]
    results = []
    for i in itertools.product(al, u, si):
        clf = MyEM(y, i)
        clf.fit()
        results.append([clf.al[-1], clf.u[-1], clf.si[-1], clf.score])  # 收集不同初值收敛的局部最优解
    best_value = max(results, key=lambda x: x[3])  # 从所有局部最优解找到相对最优解
    print("alpha:{}".format(best_value[0]))
    print("mean:{}".format(best_value[1]))
    print("std:{}".format(best_value[2]))
    # 绘制所有局部最优解直方图
    re = [res[3] for res in results]
    plt.hist(re)
    plt.show()
    end = time.time()
    print("用时：{:.2f}s".format(end - star))
