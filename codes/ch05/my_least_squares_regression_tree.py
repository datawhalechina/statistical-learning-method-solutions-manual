#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: my_least_squares_regression_tree.py
@time: 2021/8/6 21:37
@project: statistical-learning-method-solutions-manual
@desc: 习题5.2 最小二乘回归树生成算法
"""
import json

import numpy as np


# 节点类
class Node:
    def __init__(self, value, feature, left=None, right=None):
        self.value = value.tolist()
        self.feature = feature.tolist()
        self.left = left
        self.right = right

    def __repr__(self):
        return json.dumps(self, indent=3, default=lambda obj: obj.__dict__, ensure_ascii=False)


class MyLeastSquareRegTree:
    def __init__(self, train_X, y, epsilon):
        # 训练集特征值
        self.x = train_X
        # 类别
        self.y = y
        # 特征总数
        self.feature_count = train_X.shape[1]
        # 损失阈值
        self.epsilon = epsilon
        # 回归树
        self.tree = None

    def _fit(self, x, y, feature_count):
        # （1）选择最优切分点变量j与切分点s，得到选定的对(j,s)，并解得c1，c2
        (j, s, minval, c1, c2) = self._divide(x, y, feature_count)
        # 初始化树
        tree = Node(feature=j, value=x[s, j], left=None, right=None)
        # 用选定的对(j,s)划分区域，并确定响应的输出值
        if minval < self.epsilon or len(y[np.where(x[:, j] <= x[s, j])]) <= 1:
            tree.left = c1
        else:
            # 对左子区域调用步骤（1）、（2）
            tree.left = self._fit(x[np.where(x[:, j] <= x[s, j])],
                                  y[np.where(x[:, j] <= x[s, j])],
                                  self.feature_count)
        if minval < self.epsilon or len(y[np.where(x[:, j] > s)]) <= 1:
            tree.right = c2
        else:
            # 对右子区域调用步骤（1）、（2）
            tree.right = self._fit(x[np.where(x[:, j] > x[s, j])],
                                   y[np.where(x[:, j] > x[s, j])],
                                   self.feature_count)
        return tree

    def fit(self):
        self.tree = self._fit(self.x, self.y, self.feature_count)
        return self

    @staticmethod
    def _divide(x, y, feature_count):
        # 初始化损失误差
        cost = np.zeros((feature_count, len(x)))
        # 公式5.21
        for i in range(feature_count):
            for k in range(len(x)):
                # k行i列的特征值
                value = x[k, i]
                y1 = y[np.where(x[:, i] <= value)]
                c1 = np.mean(y1)
                y2 = y[np.where(x[:, i] > value)]
                if len(y2) == 0:
                    c2 = 0
                else:
                    c2 = np.mean(y2)
                y1[:] = y1[:] - c1
                y2[:] = y2[:] - c2
                cost[i, k] = np.sum(y1 * y1) + np.sum(y2 * y2)
        # 选取最优损失误差点
        cost_index = np.where(cost == np.min(cost))
        # 所选取的特征
        j = cost_index[0][0]
        # 选取特征的切分点
        s = cost_index[1][0]
        # 求两个区域的均值c1,c2
        c1 = np.mean(y[np.where(x[:, j] <= x[s, j])])
        c2 = np.mean(y[np.where(x[:, j] > x[s, j])])
        return j, s, cost[cost_index], c1, c2

    def __repr__(self):
        return str(self.tree)


if __name__ == '__main__':
    train_X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
    y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])

    model_tree = MyLeastSquareRegTree(train_X, y, epsilon=0.2)
    model_tree.fit()
    print(model_tree)
