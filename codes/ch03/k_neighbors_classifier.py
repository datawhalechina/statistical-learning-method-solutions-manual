#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: k_neighbors_classifier.py
@time: 2021/8/3 14:58
@project: statistical-learning-method-solutions-manual
@desc: 习题3.1 k近邻算法关于k值的模型比较
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = np.array([[5, 12, 1],
                 [6, 21, 0],
                 [14, 5, 0],
                 [16, 10, 0],
                 [13, 19, 0],
                 [13, 32, 1],
                 [17, 27, 1],
                 [18, 24, 1],
                 [20, 20, 0],
                 [23, 14, 1],
                 [23, 25, 1],
                 [23, 31, 1],
                 [26, 8, 0],
                 [30, 17, 1],
                 [30, 26, 1],
                 [34, 8, 0],
                 [34, 19, 1],
                 [37, 28, 1]])
# 得到特征向量
X_train = data[:, 0:2]
# 得到类别向量
y_train = data[:, 2]

# 分别构造k=1和k=2的k近邻模型
models = (KNeighborsClassifier(n_neighbors=1, n_jobs=-1),
          KNeighborsClassifier(n_neighbors=2, n_jobs=-1))
# 模型训练
models = (clf.fit(X_train, y_train) for clf in models)

# 设置图形标题
titles = ('K Neighbors with k=1',
          'K Neighbors with k=2')

# 设置图形的大小和图间距
fig = plt.figure(figsize=(15, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 分别获取第1个和第2个特征向量
X0, X1 = X_train[:, 0], X_train[:, 1]

# 得到坐标轴的最小值和最大值
x_min, x_max = X0.min() - 1, X0.max() + 1
y_min, y_max = X1.min() - 1, X1.max() + 1

# 构造网格点坐标矩阵（https://blog.csdn.net/lllxxq141592654/article/details/81532855）
# 设置0.2的目的是生成更多的网格点，数值越小，划分空间之间的分隔线越清晰
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

for clf, title, ax in zip(models, titles, fig.subplots(1, 2).flatten()):
    # 对所有网格点进行预测
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 设置颜色列表
    colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
    # 根据类别数生成颜色
    cmap = ListedColormap(colors[:len(np.unique(Z))])
    # 绘制分隔线，contourf函数用于绘制等高线，alpha表示颜色的透明度，一般设置成0.5
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)

    # 绘制样本点
    ax.scatter(X0, X1, c=y_train, s=50, edgecolors='k', cmap=cmap, alpha=0.5)

    # 计算预测准确率
    acc = clf.score(X_train, y_train)
    # 设置标题
    ax.set_title(title + ' (Accuracy: %d%%)' % (acc * 100))

plt.show()
