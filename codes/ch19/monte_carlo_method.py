#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: monte_carlo_method_demo.py
@time: 2022/7/18 10:52
@project: statistical-learning-method-solutions-manual
@desc: 习题19.1 蒙特卡洛法积分计算
"""

import numpy as np


class MonteCarloIntegration:
    def __init__(self, func_f, func_p):
        # 所求期望的函数
        self.func_f = func_f
        # 抽样分布的概率密度函数
        self.func_p = func_p

    def solve(self, num_samples):
        """
        蒙特卡罗积分法
        :param num_samples: 抽样样本数量
        :return: 样本的函数均值
        """
        samples = self.func_p(num_samples)
        vfunc_f = lambda x: self.func_f(x)
        vfunc_f = np.vectorize(vfunc_f)
        y = vfunc_f(samples)
        return np.sum(y) / num_samples


if __name__ == '__main__':
    def func_f(x):
        """定义函数f"""
        return x ** 2 * np.sqrt(2 * np.pi)


    def func_p(n):
        """定义在分布上随机抽样的函数g"""
        return np.random.standard_normal(int(n))


    # 设置样本数量
    num_samples = 1e6

    # 使用蒙特卡罗积分法进行求解
    monte_carlo_integration = MonteCarloIntegration(func_f, func_p)
    result = monte_carlo_integration.solve(num_samples)
    print("抽样样本数量:", num_samples)
    print("近似解:", result)
