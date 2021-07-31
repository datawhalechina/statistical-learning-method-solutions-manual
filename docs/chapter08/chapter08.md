# 第8章提升方法-习题

## 习题8.1
&emsp;&emsp;某公司招聘职员考查身体、业务能力、发展潜力这3项。身体分为合格1、不合格0两级，业务能力和发展潜力分为上1、中2、下3三级分类为合格1 、不合格-1两类。已知10个人的数据，如下表所示。假设弱分类器为决策树桩。试用AdaBoost算法学习一个强分类器。  

应聘人员情况数据表

&emsp;&emsp;|1|2|3|4|5|6|7|8|9|10
-|-|-|-|-|-|-|-|-|-|-
身体|0|0|1|1|1|0|1|1|1|0
业务|1|3|2|1|2|1|1|1|3|2
潜力|3|1|2|3|3|2|2|1|1|1
分类|-1|-1|-1|-1|-1|-1|1|1|-1|-1


```python
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
```

**AdaBoostClassifier分类器实现：**

采用sklearn的AdaBoostClassifier分类器直接求解，由于AdaBoostClassifier分类器默认采用CART决策树弱分类器，故不需要设置base_estimator参数。


```python
from sklearn.ensemble import AdaBoostClassifier


clf = AdaBoostClassifier()
clf.fit(X, y)
y_predict = clf.predict(X)
score = clf.score(X, y)
print("原始输出:", y)
print("预测输出:", y_predict)
print("预测正确率：{:.2%}".format(score))
```

    原始输出: [-1 -1 -1 -1 -1 -1  1  1 -1 -1]
    预测输出: [-1 -1 -1 -1 -1 -1  1  1 -1 -1]
    预测正确率：100.00%
    

**自编程实现：**


```python
# 自编程求解例8.1
import numpy as np


class AdaBoost:
    def __init__(self, X, y, tol=0.05, max_iter=10):
        # 训练数据 实例
        self.X = X
        # 训练数据 标签
        self.y = y
        # 训练中止条件 right_rate>self.tol
        self.tol = tol
        # 最大迭代次数
        self.max_iter = max_iter
        # 初始化样本权重w
        self.w = np.full((X.shape[0]), 1 / X.shape[0])
        self.G = []  # 弱分类器

    def build_stump(self):
        """
        以带权重的分类误差最小为目标，选择最佳分类阈值
        best_stump['dim'] 合适的特征所在维度
        best_stump['thresh']  合适特征的阈值
        best_stump['ineq']  树桩分类的标识lt,rt
        """
        m, n = np.shape(self.X)
        # 分类误差
        e_min = np.inf
        # 小于分类阈值的样本属于的标签类别
        sign = None
        # 最优分类树桩
        best_stump = {}
        for i in range(n):
            range_min = self.X[:, i].min()  # 求每一种特征的最大最小值
            range_max = self.X[:, i].max()
            step_size = (range_max - range_min) / n
            for j in range(-1, int(n) + 1):
                thresh_val = range_min + j * step_size
                # 计算左子树和右子树的误差
                for inequal in ['lt', 'rt']:
                    predict_vals = self.base_estimator(self.X, i, thresh_val, inequal)
                    err_arr = np.array(np.ones(m))
                    err_arr[predict_vals.T == self.y.T] = 0
                    weighted_error = np.dot(self.w, err_arr)
                    if weighted_error < e_min:
                        e_min = weighted_error
                        sign = predict_vals
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['ineq'] = inequal
        return best_stump, sign, e_min

    def updata_w(self, alpha, predict):
        """
        更新样本权重w
        """
        # 以下2行根据公式8.4 8.5 更新样本权重
        P = self.w * np.exp(-alpha * self.y * predict)
        self.w = P / P.sum()

    @staticmethod
    def base_estimator(X, dimen, threshVal, threshIneq):
        """
        计算单个弱分类器（决策树桩）预测输出
        """
        ret_array = np.ones(np.shape(X)[0])  # 预测矩阵
        # 左叶子 ，整个矩阵的样本进行比较赋值
        if threshIneq == 'lt':
            ret_array[X[:, dimen] <= threshVal] = -1.0
        else:
            ret_array[X[:, dimen] > threshVal] = -1.0
        return ret_array

    def fit(self):
        """
        对训练数据进行学习
        """
        G = 0
        for i in range(self.max_iter):
            best_stump, sign, error = self.build_stump()  # 获取当前迭代最佳分类阈值
            alpha = 1 / 2 * np.log((1 - error) / error)  # 计算本轮弱分类器的系数
            # 弱分类器权重
            best_stump['alpha'] = alpha
            # 保存弱分类器
            self.G.append(best_stump)
            # 以下3行计算当前总分类器（之前所有弱分类器加权和）分类效率
            G += alpha * sign
            y_predict = np.sign(G)
            error_rate = np.sum(np.abs(y_predict - self.y)) / 2 / self.y.shape[0]
            if error_rate < self.tol:  # 满足中止条件 则跳出循环
                print("迭代次数:", i + 1)
                break
            else:
                self.updata_w(alpha, y_predict)  # 若不满足，更新权重，继续迭代

    def predict(self, X):
        """
        对新数据进行预测
        """
        m = np.shape(X)[0]
        G = np.zeros(m)
        for i in range(len(self.G)):
            stump = self.G[i]
            # 遍历每一个弱分类器，进行加权
            _G = self.base_estimator(X, stump['dim'], stump['thresh'], stump['ineq'])
            alpha = stump['alpha']
            G += alpha * _G
        y_predict = np.sign(G)
        return y_predict.astype(int)

    def score(self, X, y):
        """对训练效果进行评价"""
        y_predict = self.predict(X)
        error_rate = np.sum(np.abs(y_predict - y)) / 2 / y.shape[0]
        return 1 - error_rate
```


```python
clf = AdaBoost(X, y)
clf.fit()
y_predict = clf.predict(X)
score = clf.score(X, y)
print("原始输出:", y)
print("预测输出:", y_predict)
print("预测正确率：{:.2%}".format(score))
```

    迭代次数: 8
    原始输出: [-1 -1 -1 -1 -1 -1  1  1 -1 -1]
    预测输出: [-1 -1 -1 -1 -1 -1  1  1 -1 -1]
    预测正确率：100.00%
    

## 习题8.2
&emsp;&emsp;比较支持向量机、 AdaBoost 、Logistic回归模型的学习策略与算法

**解答：**  
- **支持向量机**  
学习策略：极小化正则化合页损失，软间隔最大化；  
学习算法：序列最小最优化算法（SMO）  
- **AdaBoost**  
学习策略：极小化加法模型指数损失；  
学习算法：前向分步加法算法  
- **Logistic回归**  
学习策略：极大似然估计，正则化的极大似然估计；  
学习算法：改进的迭代尺度算法，梯度下降，拟牛顿法
