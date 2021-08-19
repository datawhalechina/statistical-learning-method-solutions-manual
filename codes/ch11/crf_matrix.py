#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: crf_matrix.py
@time: 2021/8/19 2:15
@project: statistical-learning-method-solutions-manual
@desc: 习题11.4 使用条件随机场矩阵形式，计算所有路径状态序列的概率及概率最大的状态序列
"""

import numpy as np


class CRFMatrix:
    def __init__(self, M, start, stop):
        # 随机矩阵
        self.M = M
        #
        self.start = start
        self.stop = stop
        self.path_prob = None

    def _create_path(self):
        """按照图11.6的状态路径图，生成路径"""
        # 初始化start结点
        path = [self.start]
        for i in range(1, len(self.M)):
            paths = []
            for _, r in enumerate(path):
                temp = np.transpose(r)
                # 添加状态结点1
                paths.append(np.append(temp, 1))
                # 添加状态结点2
                paths.append(np.append(temp, 2))
            path = paths.copy()

        # 添加stop结点
        path = [np.append(r, self.stop) for _, r in enumerate(path)]
        return path

    def fit(self):
        path = self._create_path()
        pr = []
        for _, row in enumerate(path):
            p = 1
            for i in range(len(row) - 1):
                a = row[i]
                b = row[i + 1]
                # 根据公式11.24，计算条件概率
                p *= M[i][a - 1][b - 1]
            pr.append((row.tolist(), p))
        # 按照概率从大到小排列
        pr = sorted(pr, key=lambda x: x[1], reverse=True)
        self.path_prob = pr

    def print(self):
        # 打印结果
        print("以start=%s为起点stop=%s为终点的所有路径的状态序列y的概率为：" % (self.start, self.stop))
        for path, p in self.path_prob:
            print("    路径为：" + "->".join([str(x) for x in path]), end=" ")
            print("概率为：" + str(p))
        print("概率最大[" + str(self.path_prob[0][1]) + "]的状态序列为:",
              "->".join([str(x) for x in self.path_prob[0][0]]))


if __name__ == '__main__':
    # 创建随机矩阵
    M1 = [[0, 0], [0.5, 0.5]]
    M2 = [[0.3, 0.7], [0.7, 0.3]]
    M3 = [[0.5, 0.5], [0.6, 0.4]]
    M4 = [[0, 1], [0, 1]]
    M = [M1, M2, M3, M4]
    # 构建条件随机场的矩阵模型
    crf = CRFMatrix(M=M, start=2, stop=2)
    # 得到所有路径的状态序列的概率
    crf.fit()
    # 打印结果
    crf.print()
