#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: my_decision_tree.py
@time: 2021/8/5 17:11
@project: statistical-learning-method-solutions-manual
@desc: 习题5.1 自编程实现C4.5生成算法
"""
import json
from collections import Counter

import numpy as np


# 节点类
class Node:
    def __init__(self, node_type, class_name, feature_name=None,
                 info_gain_ratio_value=0.0):
        # 结点类型（internal或leaf）
        self.node_type = node_type
        # 特征名
        self.feature_name = feature_name
        # 类别名
        self.class_name = class_name
        # 子结点树
        self.child_nodes = []
        # Gini指数值
        self.info_gain_ratio_value = info_gain_ratio_value

    def __repr__(self):
        return json.dumps(self, indent=3, default=lambda obj: obj.__dict__, ensure_ascii=False)

    def add_sub_tree(self, key, sub_tree):
        self.child_nodes.append({"condition": key, "sub_tree": sub_tree})


class MyDecisionTree:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.tree = None

    def fit(self, train_set, y, feature_names):
        features_indices = list(range(len(feature_names)))
        self.tree = self._fit(train_set, y, features_indices, feature_names)
        return self

    # C4.5算法
    def _fit(self, train_data, y, features_indices, feature_labels):
        LEAF = 'leaf'
        INTERNAL = 'internal'
        class_num = len(np.unique(y))

        # （1）如果训练数据集所有实例都属于同一类Ck
        label_set = set(y)
        if len(label_set) == 1:
            # 将Ck作为该结点的类
            return Node(node_type=LEAF, class_name=label_set.pop())

        # （2）如果特征集为空
        # 计算每一个类出现的个数
        class_len = Counter(y).most_common()
        (max_class, max_len) = class_len[0]

        if len(features_indices) == 0:
            # 将实例数最大的类Ck作为该结点的类
            return Node(LEAF, class_name=max_class)

        # （3）按式(5.10)计算信息增益，并选择信息增益最大的特征
        max_feature = 0
        max_gda = 0
        D = y.copy()
        # 计算特征集A中各特征
        for feature in features_indices:
            # 选择训练集中的第feature列（即第feature个特征）
            A = np.array(train_data[:, feature].flat)
            # 计算信息增益
            gda = self._calc_ent_grap(A, D)
            if self._calc_ent(A) != 0:
                # 计算信息增益比
                gda /= self._calc_ent(A)
            # 选择信息增益最大的特征Ag
            if gda > max_gda:
                max_gda, max_feature = gda, feature

        # （4）如果Ag信息增益小于阈值
        if max_gda < self.epsilon:
            # 将训练集中实例数最大的类Ck作为该结点的类
            return Node(LEAF, class_name=max_class)

        max_feature_label = feature_labels[max_feature]

        # （6）移除已选特征Ag
        sub_feature_indecs = np.setdiff1d(features_indices, max_feature)
        sub_feature_labels = np.setdiff1d(feature_labels, max_feature_label)

        # （5）构建非空子集
        # 构建结点
        feature_name = feature_labels[max_feature]
        tree = Node(INTERNAL, class_name=None, feature_name=feature_name,
                    info_gain_ratio_value=max_gda)

        max_feature_col = np.array(train_data[:, max_feature].flat)
        # 将类按照对应的实例数递减顺序排列
        feature_value_list = [x[0] for x in Counter(max_feature_col).most_common()]
        # 遍历Ag的每一个可能值ai
        for feature_value in feature_value_list:
            index = []
            for i in range(len(y)):
                if train_data[i][max_feature] == feature_value:
                    index.append(i)

            # 递归调用步（1）~步（5），得到子树
            sub_train_set = train_data[index]
            sub_train_label = y[index]
            sub_tree = self._fit(sub_train_set, sub_train_label, sub_feature_indecs, sub_feature_labels)
            # 在结点中，添加其子结点构成的树
            tree.add_sub_tree(feature_value, sub_tree)

        return tree

    # 计算数据集x的经验熵H(x)
    @staticmethod
    def _calc_ent(x):
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            p = float(x[x == x_value].shape[0]) / x.shape[0]
            logp = np.log2(p)
            ent -= p * logp

        return ent

    # 计算条件熵H(y/x)
    def _calc_condition_ent(self, x, y):
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            sub_y = y[x == x_value]
            temp_ent = self._calc_ent(sub_y)
            ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

        return ent

    # 计算信息增益
    def _calc_ent_grap(self, x, y):
        base_ent = self._calc_ent(y)
        condition_ent = self._calc_condition_ent(x, y)
        ent_grap = base_ent - condition_ent

        return ent_grap

    def __repr__(self):
        return str(self.tree)


if __name__ == '__main__':
    # 表5.1的训练数据集
    feature_names = np.array(["年龄", "有工作", "有自己的房子", "信贷情况"])
    X_train = np.array([
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "是", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "否", "否", "好"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"]
    ])
    y = np.array(["否", "否", "是", "是", "否",
                  "否", "否", "是", "是", "是",
                  "是", "是", "是", "是", "否"])

    dt_tree = MyDecisionTree(epsilon=0.1)
    dt_tree.fit(X_train, y, feature_names)
    print(dt_tree)
