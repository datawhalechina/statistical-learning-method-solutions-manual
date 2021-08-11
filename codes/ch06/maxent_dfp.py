#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: maxent_dfp.py
@time: 2021/8/10 2:28
@project: statistical-learning-method-solutions-manual
@desc: 习题6.3 最大熵模型学习的DFP算法
"""
import copy
from collections import defaultdict

import numpy as np
from scipy.optimize import fminbound


class MaxEntDFP:
    def __init__(self, epsilon, max_iter=1000, distance=0.01):
        """
        最大熵的DFP算法
        :param epsilon: 迭代停止阈值
        :param max_iter: 最大迭代次数
        :param distance: 一维搜索的长度范围
        """
        self.distance = distance
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.w = None
        self._dataset_X = None
        self._dataset_y = None
        # 标签集合，相当去去重后的y
        self._y = set()
        # key为(x,y), value为对应的索引号ID
        self._xyID = {}
        # key为对应的索引号ID, value为(x,y)
        self._IDxy = {}
        # 经验分布p(x,y)
        self._pxy_dic = defaultdict(int)
        # 样本数
        self._N = 0
        # 特征键值(x,y)的个数
        self._n = 0
        # 实际迭代次数
        self.n_iter_ = 0

    # 初始化参数
    def init_params(self, X, y):
        self._dataset_X = copy.deepcopy(X)
        self._dataset_y = copy.deepcopy(y)
        self._N = X.shape[0]

        for i in range(self._N):
            xi, yi = X[i], y[i]
            self._y.add(yi)
            for _x in xi:
                self._pxy_dic[(_x, yi)] += 1

        self._n = len(self._pxy_dic)
        # 初始化权重w
        self.w = np.zeros(self._n)

        for i, xy in enumerate(self._pxy_dic):
            self._pxy_dic[xy] /= self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

    def calc_zw(self, X, w):
        """书中第100页公式6.23，计算Zw(x)"""
        zw = 0.0
        for y in self._y:
            zw += self.calc_ewf(X, y, w)
        return zw

    def calc_ewf(self, X, y, w):
        """书中第100页公式6.22，计算分子"""
        sum_wf = self.calc_wf(X, y, w)
        return np.exp(sum_wf)

    def calc_wf(self, X, y, w):
        sum_wf = 0.0
        for x in X:
            if (x, y) in self._pxy_dic:
                sum_wf += w[self._xyID[(x, y)]]
        return sum_wf

    def calc_pw_yx(self, X, y, w):
        """计算Pw(y|x)"""
        return self.calc_ewf(X, y, w) / self.calc_zw(X, w)

    def calc_f(self, w):
        """计算f(w)"""
        fw = 0.0
        for i in range(self._n):
            x, y = self._IDxy[i]
            for dataset_X in self._dataset_X:
                if x not in dataset_X:
                    continue
                fw += np.log(self.calc_zw(x, w)) - self._pxy_dic[(x, y)] * self.calc_wf(dataset_X, y, w)

        return fw

    # DFP算法
    def fit(self, X, y):
        self.init_params(X, y)

        def calc_dfw(i, w):
            """计算书中第107页的拟牛顿法f(w)的偏导"""

            def calc_ewp(i, w):
                """计算偏导左边的公式"""
                ep = 0.0
                x, y = self._IDxy[i]
                for dataset_X in self._dataset_X:
                    if x not in dataset_X:
                        continue
                    ep += self.calc_pw_yx(dataset_X, y, w) / self._N
                return ep

            def calc_ep(i):
                """计算关于经验分布P(x,y)的期望值"""
                (x, y) = self._IDxy[i]
                return self._pxy_dic[(x, y)]

            return calc_ewp(i, w) - calc_ep(i)

        # 算出g(w)，是n*1维矩阵
        def calc_gw(w):
            return np.array([[calc_dfw(i, w) for i in range(self._n)]]).T

        # （1）初始正定对称矩阵，单位矩阵
        Gk = np.array(np.eye(len(self.w), dtype=float))

        # （2）计算g(w0)
        w = self.w
        gk = calc_gw(w)
        # 判断gk的范数是否小于阈值
        if np.linalg.norm(gk, ord=2) < self.epsilon:
            self.w = w
            return

        k = 0
        for _ in range(self.max_iter):
            # （3）计算pk
            pk = -Gk.dot(gk)

            # 梯度方向的一维函数
            def _f(x):
                z = w + np.dot(x, pk).T[0]
                return self.calc_f(z)

            # （4）进行一维搜索，找到使得函数最小的lambda
            _lambda = fminbound(_f, -self.distance, self.distance)

            delta_k = _lambda * pk
            # （5）更新权重
            w += delta_k.T[0]

            # （6）计算gk+1
            gk1 = calc_gw(w)
            # 判断gk1的范数是否小于阈值
            if np.linalg.norm(gk1, ord=2) < self.epsilon:
                self.w = w
                break
            # 根据DFP算法的迭代公式（附录B.24公式）计算Gk
            yk = gk1 - gk
            Pk = delta_k.dot(delta_k.T) / (delta_k.T.dot(yk))
            Qk = Gk.dot(yk).dot(yk.T).dot(Gk) / (yk.T.dot(Gk).dot(yk)) * (-1)
            Gk = Gk + Pk + Qk
            gk = gk1

            # （7）置k=k+1
            k += 1

        self.w = w
        self.n_iter_ = k

    def predict(self, x):
        result = {}
        for y in self._y:
            prob = self.calc_pw_yx(x, y, self.w)
            result[y] = prob

        return result


if __name__ == '__main__':
    # 训练数据集
    dataset = np.array([['no', 'sunny', 'hot', 'high', 'FALSE'],
                        ['no', 'sunny', 'hot', 'high', 'TRUE'],
                        ['yes', 'overcast', 'hot', 'high', 'FALSE'],
                        ['yes', 'rainy', 'mild', 'high', 'FALSE'],
                        ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
                        ['no', 'rainy', 'cool', 'normal', 'TRUE'],
                        ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
                        ['no', 'sunny', 'mild', 'high', 'FALSE'],
                        ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
                        ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
                        ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
                        ['yes', 'overcast', 'mild', 'high', 'TRUE'],
                        ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
                        ['no', 'rainy', 'mild', 'high', 'TRUE']])

    X_train = dataset[:, 1:]
    y_train = dataset[:, 0]

    mae = MaxEntDFP(epsilon=1e-4, max_iter=1000, distance=0.01)
    mae.fit(X_train, y_train)
    print("模型训练迭代次数：{}次".format(mae.n_iter_))
    print("模型权重：{}".format(mae.w))

    result = mae.predict(['overcast', 'mild', 'high', 'FALSE'])
    print("预测结果：", result)
