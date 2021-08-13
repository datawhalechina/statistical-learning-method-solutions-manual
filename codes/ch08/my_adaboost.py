#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: my_adaboost.py
@time: 2021/8/13 21:17
@project: statistical-learning-method-solutions-manual
@desc: 习题8.1 自编程实现AdaBoost算法
"""

import numpy as np


class MyAdaBoost:
    def __init__(self, tol=0.05, max_iter=10):
        # 特征
        self.X = None
        # 标签
        self.y = None
        # 分类误差小于精度时，分类器训练中止
        self.tol = tol
        # 最大迭代次数
        self.max_iter = max_iter
        # 权值分布
        self.w = None
        # 弱分类器集合
        self.G = []

    def build_stump(self):
        """
        以带权重的分类误差最小为目标，选择最佳分类阈值，得到最佳的决策树桩
        best_stump['dim'] 合适特征的所在维度
        best_stump['thresh']  合适特征的阈值
        best_stump['ineq']  树桩分类的标识lt,rt
        """
        m, n = np.shape(self.X)
        # 分类误差
        min_error = np.inf
        # 小于分类阈值的样本所属的标签类别
        sign = None
        # 最优决策树桩
        best_stump = {}
        for i in range(n):
            # 求每一种特征的最小值和最大值
            range_min = self.X[:, i].min()
            range_max = self.X[:, i].max()
            step_size = (range_max - range_min) / n
            for j in range(-1, int(n) + 1):
                # 根据n的值，构造切分点
                thresh_val = range_min + j * step_size
                # 计算左子树和右子树的误差
                for inequal in ['lt', 'rt']:
                    # (a)得到基本分类器
                    predict_values = self.base_estimator(self.X, i, thresh_val, inequal)
                    # (b)计算在训练集上的分类误差率
                    err_arr = np.array(np.ones(m))
                    err_arr[predict_values.T == self.y.T] = 0
                    weighted_error = np.dot(self.w, err_arr)
                    if weighted_error < min_error:
                        min_error = weighted_error
                        sign = predict_values
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['ineq'] = inequal
        return best_stump, sign, min_error

    def updata_w(self, alpha, predict):
        """
        更新样本权重w
        :param alpha: alpha
        :param predict: yi
        :return:
        """
        # (d)根据迭代公式，更新权值分布
        P = self.w * np.exp(-alpha * self.y * predict)
        self.w = P / P.sum()

    @staticmethod
    def base_estimator(X, dimen, thresh_val, thresh_ineq):
        """
        计算单个弱分类器（决策树桩）预测输出
        :param X: 特征
        :param dimen: 特征的位置（即第几个特征）
        :param thresh_val: 切分点
        :param thresh_ineq: 标记结点的位置，可取左子树(lt)，右子树(rt)
        :return: 返回预测结果矩阵
        """
        # 预测结果矩阵
        ret_array = np.ones(np.shape(X)[0])
        # 左叶子 ，整个矩阵的样本进行比较赋值
        if thresh_ineq == 'lt':
            ret_array[X[:, dimen] >= thresh_val] = -1.0
        else:
            ret_array[X[:, dimen] < thresh_val] = -1.0
        return ret_array

    def fit(self, X, y):
        """
        对分类器进行训练
        """
        self.X = X
        self.y = y
        # （1）初始化训练数据的权值分布
        self.w = np.full((X.shape[0]), 1 / X.shape[0])
        G = 0
        # （2）对m=1,2,...,M进行遍历
        for i in range(self.max_iter):
            # (b)得到Gm(x)的分类误差error，获取当前迭代最佳分类阈值sign
            best_stump, sign, error = self.build_stump()
            # (c)计算弱分类器Gm(x)的系数
            alpha = 1 / 2 * np.log((1 - error) / error)
            # 弱分类器Gm(x)权重
            best_stump['alpha'] = alpha
            # 保存弱分类器Gm(x)，得到分类器集合G
            self.G.append(best_stump)
            # 计算当前总分类器（之前所有弱分类器加权和）误差率
            G += alpha * sign
            y_predict = np.sign(G)
            error_rate = np.sum(np.abs(y_predict - self.y)) / 2 / self.y.shape[0]
            if error_rate < self.tol:
                # 满足中止条件，则跳出循环
                print("迭代次数：{}次".format(i + 1))
                break
            else:
                # (d)更新训练数据集的权值分布
                self.updata_w(alpha, y_predict)

    def predict(self, X):
        """对新数据进行预测"""
        m = np.shape(X)[0]
        G = np.zeros(m)
        for i in range(len(self.G)):
            stump = self.G[i]
            # 遍历每一个弱分类器，进行加权
            _G = self.base_estimator(X, stump['dim'], stump['thresh'], stump['ineq'])
            alpha = stump['alpha']
            # (3)构建基本分类器的线性组合
            G += alpha * _G
        # 计算最终分类器的预测结果
        y_predict = np.sign(G)
        return y_predict.astype(int)

    def score(self, X, y):
        """计算分类器的预测准确率"""
        y_predict = self.predict(X)
        error_rate = np.sum(np.abs(y_predict - y)) / 2 / y.shape[0]
        return 1 - error_rate

    def print_G(self):
        i = 1
        s = "G(x) = sign[f(x)] = sign["
        for stump in self.G:
            if i != 1:
                s += " + "
            s += "{}·G{}(x)".format(round(stump['alpha'], 4), i)
            i += 1
        s += "]"
        return s


if __name__ == '__main__':
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

    clf = MyAdaBoost()
    clf.fit(X, y)
    y_predict = clf.predict(X)
    score = clf.score(X, y)
    print("原始输出:", y)
    print("预测输出:", y_predict)
    print("预测正确率：{:.2%}".format(score))
    print("最终分类器G(x)为:", clf.print_G())
