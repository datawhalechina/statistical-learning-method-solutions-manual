#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: gmm_demo.py
@time: 2021/8/14 16:04
@project: statistical-learning-method-solutions-manual
@desc: 习题9.3 使用GaussianMixture求解两个分量高斯混合模型的6个参数
"""

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# 初始化观测数据
data = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]).reshape(-1, 1)

# 设置n_components=2，表示两个分量高斯混合模型
gmm_model = GaussianMixture(n_components=2)
# 对模型进行参数估计
gmm_model.fit(data)
# 对数据进行聚类
labels = gmm_model.predict(data)

# 得到分类结果
print("分类结果：labels = {}\n".format(labels))
print("两个分量高斯混合模型的6个参数如下：")
# 得到参数u1,u2
print("means =", gmm_model.means_.reshape(1, -1))
# 得到参数sigma1, sigma1
print("covariances =", gmm_model.covariances_.reshape(1, -1))
# 得到参数a1, a2
print("weights = ", gmm_model.weights_.reshape(1, -1))

# 绘制观测数据的聚类情况
for i in range(0, len(labels)):
    if labels[i] == 0:
        plt.scatter(i, data.take(i), s=15, c='red')
    elif labels[i] == 1:
        plt.scatter(i, data.take(i), s=15, c='blue')
plt.title('Gaussian Mixture Model')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
