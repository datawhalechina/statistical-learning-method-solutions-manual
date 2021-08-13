#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: adaboost_demo.py
@time: 2021/8/13 17:16
@project: statistical-learning-method-solutions-manual
@desc: 习题8.1 使用sklearn的AdaBoostClassifier分类器实现
"""

from sklearn.ensemble import AdaBoostClassifier
import numpy as np

# 加载训练数据
X = np.array([[0, 1, 3],
              [0, 3, 1],
              [1, 2, 2],
              [1, 1, 3],
              [1, 2, 3],
              [0, 1, 2],
              [1, 1, 2],
              [1, 1, 1],
              [1, 3, 1],
              [0, 2, 1]
              ])
y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])

# 使用sklearn的AdaBoostClassifier分类器
clf = AdaBoostClassifier()
# 进行分类器训练
clf.fit(X, y)
# 对数据进行预测
y_predict = clf.predict(X)
# 得到分类器的预测准确率
score = clf.score(X, y)
print("原始输出:", y)
print("预测输出:", y_predict)
print("预测准确率：{:.2%}".format(score))
