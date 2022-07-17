#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: divisive_clustering.py
@time: 2022/7/17 0:42
@project: statistical-learning-method-solutions-manual
@desc: 习题15.1 分裂聚类算法
"""
import numpy as np


class DivisiveClustering:
    def __init__(self, num_class):
        # 聚类类别个数
        self.num_class = num_class
        # 聚类数据集
        self.cluster_data = []
        if num_class > 1:
            self.cluster_data = [[] for _ in range(num_class)]

    def fit(self, data):
        """
        :param data: 数据集
        """
        num_sample = data.shape[0]

        if self.num_class == 1:
            # 如果只设定了一类，将所有数据放入到该类中
            for d in data:
                self.cluster_data.append(d)
        else:
            # (1) 构造1个类，该类包含全部样本
            # 初始化类中心
            class_center = []

            # (2) 计算n个样本两两之间的欧氏距离
            distance = np.zeros((num_sample, num_sample))
            for i in range(num_sample):
                for j in range(i + 1, num_sample):
                    distance[j, i] = distance[i, j] = np.linalg.norm(data[i, :] - data[j, :], ord=2)

            # (3) 分裂距离最大的两个样本，并设置为各自的类中心
            index = np.where(np.max(distance) == distance)
            class_1 = index[1][0]
            class_2 = index[1][1]
            # 记录已经分裂完成的样本
            finished_data = [class_1, class_2]
            class_center.append(data[class_1, :])
            class_center.append(data[class_2, :])

            num_class_temp = 2
            # (5)判断类的个数是否满足设定的样本类别数
            while num_class_temp != self.num_class:
                # (4.1)计算剩余样本与目前各个类中心的距离
                data2class_distance = np.zeros((num_sample, 1))
                for i in range(num_sample):
                    # 计算样本到各类中心的距离总和
                    data2class_sum = 0
                    for j, center_data in enumerate(class_center):
                        if i not in finished_data:
                            data2class_sum += np.linalg.norm(data[i, :] - center_data)
                    data2class_distance[i] = data2class_sum

                # (4.2) 分裂类间距离最大的样本作为新的类中心，构造一个新类
                class_new_index = np.argmax(data2class_distance)
                num_class_temp += 1
                finished_data.append(class_new_index)
                # 添加到类中心集合中
                class_center.append(data[class_new_index, :])

            # 根据当前的类中心，按照最近邻的原则对整个数据集进行分类
            for i in range(num_sample):
                data2class_distance = []
                for j, center_data in enumerate(class_center):
                    # 计算每个样本到类中心的距离
                    data2class_distance.append(np.linalg.norm(data[i, :] - center_data))

                # 将样本划分到最近的中心的类中
                label = np.argmin(data2class_distance)
                self.cluster_data[label].append(data[i, :])


if __name__ == '__main__':
    # 使用书中例14.2的样本数据集
    dataset = np.array([[0, 2],
                        [0, 0],
                        [1, 0],
                        [5, 0],
                        [5, 2]])

    num_class = 2
    divi_cluster = DivisiveClustering(num_class=num_class)
    divi_cluster.fit(dataset)
    print("分类数:", num_class)
    for i in range(num_class):
        print(f"class_{i}:", divi_cluster.cluster_data[i])
