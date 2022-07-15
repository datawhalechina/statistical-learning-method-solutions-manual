#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: outer_product_expansion.py
@time: 2022/7/15 16:09
@project: statistical-learning-method-solutions-manual
@desc: 习题15.2 外积展开式
"""

import numpy as np

A = np.array([[2, 4],
              [1, 3],
              [0, 0],
              [0, 0]])

# 调用numpy的svd方法
U, S, V = np.linalg.svd(A)
np.set_printoptions()

print("U=", U)
print("S=", S)
print("V=", V.T)

calc = S[0] * np.outer(U[:, 0], V[:, 0]) + S[1] * np.outer(U[:, 1], V[:, 1])
print("A=", calc)


