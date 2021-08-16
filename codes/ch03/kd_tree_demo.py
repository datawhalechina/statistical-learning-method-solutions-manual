#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: kd_tree_demo.py
@time: 2021/8/3 17:08
@project: statistical-learning-method-solutions-manual
@desc: 习题3.2 kd树的构建与求最近邻点
"""

import numpy as np
from sklearn.neighbors import KDTree

# 构造例题3.2的数据集
train_data = np.array([[2, 3],
                       [5, 4],
                       [9, 6],
                       [4, 7],
                       [8, 1],
                       [7, 2]])
# （1）使用sklearn的KDTree类，构建平衡kd树
# 设置leaf_size为2，表示平衡树
tree = KDTree(train_data, leaf_size=2)

# （2）使用tree.query方法，设置k=1，查找(3, 4.5)的最近邻点
# dist表示与最近邻点的距离，ind表示最近邻点在train_data的位置
dist, ind = tree.query(np.array([[3, 4.5]]), k=1)
node_index = ind[0]

# 得到最近邻点
x1 = train_data[node_index][0][0]
x2 = train_data[node_index][0][1]
print("x点的最近邻点是({0}, {1})".format(x1, x2))
