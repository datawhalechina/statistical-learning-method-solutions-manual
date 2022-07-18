# 第8章 提升方法

## 习题8.1
&emsp;&emsp;某公司招聘职员考查身体、业务能力、发展潜力这3项。身体分为合格1、不合格0两级，业务能力和发展潜力分为上1、中2、下3三级。分类为合格1 、不合格-1两类。已知10个人的数据，如下表所示。假设弱分类器为决策树桩。试用AdaBoost算法学习一个强分类器。  

应聘人员情况数据表

&emsp;&emsp;|1|2|3|4|5|6|7|8|9|10
-|-|-|-|-|-|-|-|-|-|-
身体|0|0|1|1|1|0|1|1|1|0
业务|1|3|2|1|2|1|1|1|3|2
潜力|3|1|2|3|3|2|2|1|1|1
分类|-1|-1|-1|-1|-1|-1|1|1|-1|-1

**解答：**

**解答思路：**
1. 列出AdaBoost算法；
2. 采用sklearn的AdaBoostClassifier分类器，构建并训练得到强分类器；
3. 自编程实现AdaBoost算法，并训练得到强分类器。

**解答步骤：**

**第1步：提升方法AdaBoost算法**

&emsp;&emsp;根据书中第156页算法8.1：
> 输入：训练数据集$T=\{(x_1,y_1), (x_2,y_2), \cdots, (x_N,y_N)\}$，其中$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} = \{-1, +1\}$；弱学习算法；  
输出：最终分类器$G(x)$。  
（1）初始化训练数据的权值分布
> $$
D_1 = (w_{11}, \cdots, w_{1i}, \cdots, w_{1N}), \quad w_{1i} = \frac{1}{N}, \quad i = 1, 2, \cdots, N
$$  
>（2）对$m=1, 2, \cdots, M$  
&emsp;&emsp;（a）使用具有权值分布$D_m$的训练数据集学习，得到基本分类器
> $$
G_m(x):\mathcal{X} \rightarrow \{-1, +1\}
$$
> &emsp;&emsp;（b）计算$G_m(x)$在训练数据集上的分类误差率
> $$
e_m = \sum_{i=1}^N P(G_m(x_i) \neq y_i) = \sum_{i=1}^N w_{mi}I(G_m(x_i) \neq y_i)
$$
> &emsp;&emsp;（c）计算$G_m(x)$的系数
> $$
\alpha_m = \frac{1}{2} \log \frac{1 - e_m}{e_m}
$$
> &emsp;&emsp;这里的对数是自然对数。  
&emsp;&emsp;（d）更新训练数据集的权值分布
> $$
D_{m+1} = (w_{m+1,1}, \cdots, w_{m+1, i}, \cdots, w_{m+1,N}) \\
w_{m+1,i} = \frac{w_{mi}}{Z_m} \exp(-\alpha_m y_i G_m(x_i)), \quad i = 1, 2, \cdots, N
$$
> &emsp;&emsp;这里，$Z_m$是规范化因子
> $$
Z_m = \sum_{i=1}^N w_{mi} \exp(-\alpha_m y_i G_m(x_i))
$$
> &emsp;&emsp;它使$D_{m+1}$成为一个概率分布。  
（3）构建基本分类器的线性组合
> $$
f(x) = \sum_{m=1}^M \alpha_m G_m(x)
$$
> 得到最终分类器
> $$
\begin{aligned}
G(x) &= \text{sign}(f(x)) \\
&= \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x) \right)
\end{aligned}
$$

**第2步：采用AdaBoostClassifier分类器实现**

&emsp;&emsp;根据题目要求弱分类器采用决策树，通过sklearn的AdaBoostClassifier类，构建分类器，由于AdaBoostClassifier分类器默认采用CART决策树弱分类器，故不需要设置base_estimator参数。


```python
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
```

    原始输出: [-1 -1 -1 -1 -1 -1  1  1 -1 -1]
    预测输出: [-1 -1 -1 -1 -1 -1  1  1 -1 -1]
    预测准确率：100.00%
    

**第3步：自编程实现AdaBoost算法**

代码思路：
1. 写出fit函数，即分类器训练函数；
2. 根据书中第158页例8.1，编写build_stump函数，用于得到分类误差最低的基本分类器；
3. 根据算法第2步(a)~(c)，编写代码；
4. 根据算法第2步(d)，编写updata_w函数，用于更新训练数据集的权值分布；
5. 编写predict函数，用于预测数据；
6. 【附加】编写score函数，用于计算分类器的预测准确率；
7. 【附加】编写print_G函数，用于打印最终分类器。


```python
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
```


```python
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
```

    迭代次数：8次
    原始输出: [-1 -1 -1 -1 -1 -1  1  1 -1 -1]
    预测输出: [-1 -1 -1 -1 -1 -1  1  1 -1 -1]
    预测正确率：100.00%
    最终分类器G(x)为: G(x) = sign[f(x)] = sign[0.6931·G1(x) + 0.7332·G2(x) + 0.4993·G3(x) + 0.6236·G4(x) + 0.7214·G5(x) + 0.5575·G6(x) + 0.6021·G7(x) + 0.8397·G8(x)]
    

## 习题8.2
&emsp;&emsp;比较支持向量机、 AdaBoost 、Logistic回归模型的学习策略与算法

**解答：**

**解答思路：**
1. 列出支持向量机的学习策略与学习算法
2. 列出AdaBoost的学习策略与学习算法
3. 列出Logistic回归模型的学习策略与学习算法
4. 比较三者的学习策略与算法

**解答步骤：**

**第1步：支持向量机的学习策略与算法**

&emsp;&emsp;根据书中第131页7.2.4节（合页损失函数）
> &emsp;&emsp;对于线性支持向量机学习来说，其模型为分离超平面$w^* \cdot x + b^* = 0$及决策函数$f(x)=\text{sign}(w^* \cdot x + b^*)$，其学习策略为软间隔最大化，学习算法为凸二次规划。  
> &emsp;&emsp;线性支持向量机学习还有另外一种解释，就是最小化一下目标函数：
> $$
\sum_{i=1}^N [1 - y_i(w \cdot x_i + b)]_+ + \lambda \|w\|^2
$$
> 目标函数的第1项是经验损失或经验风险，函数
> $$
L(y(w \cdot b + x)) = [1 - y_i(w \cdot x_i + b)]_+
$$
> 被称为合页损失函数，第2项是系数为$\lambda$的$w$的$L_2$范数，是正则化项。

&emsp;&emsp;根据书中第142~143页7.4节（序列最小最优化算法）
> &emsp;&emsp;SMO算法是一种启发式算法，其基本思路是：如果所有变量的解都满足此最优化问题的KKT条件，那么这个最优化问题的解就得到了。因为KKT条件是该最优化问题的充分必要条件。  
> &emsp;&emsp;整个SMO算法包括两个部分：求解两个变量二次规划的解析方法和选择变量的启发式方法。

综上所述：
1. 支持向量机的学习策略：软间隔最大化、最小化由合页损失函数和正则化项组成的目标函数
2. 支持向量机的学习算法：凸二次规划、SMO算法（序列最小最优化算法）

**第2步：AdaBoost的学习策略与算法**

&emsp;&emsp;根据书中第162页8.3节（AdbBoost算法的解释）
> &emsp;&emsp;AdaBoost算法还有另一个解释，即可认为AdaBoost算法是模型为加法模型、损失函数为指数函数、学习算法为前向分步算法时的二类分类学习方法。  
> &emsp;&emsp;给定训练数据及损失函数$L(y,f(x))$的条件下，学习加法模型$f(x)$成为经验风险极小化即损失函数极小化问题：
> $$
\min \limits_{\beta_m ,\gamma_m} \sum_{i=1}^N L \left(y_i, \sum_{m=1}^M \beta_m b(x_i;\gamma_m) \right)
$$
>
> &emsp;&emsp;定理8.3 AdaBoost算法是前向分步加法算法的特例。这时，模型是由基本分类器组成的加法模型，损失函数是指数函数。

综上所述：
1. AdaBoost的学习策略：极小化通过加法模型组成的指数损失函数
2. AdaBoost的学习算法：学习加法模型的前向分步算法

**第3步：Logistic回归模型的学习策略与算法**

&emsp;&emsp;根据书中第93页6.1.3节（模型参数估计）
> &emsp;&emsp;Logistic回归模型学习时，对于给定的训练数据集$T=\{(x_1,y_1), (x_2,y_2), \cdots, (x_N,y_N)\}$，其中$x_i \in R^n$，$y_i \in \{0, 1\}$，可以应用极大似然估计法估计模型参数，从而得到Logistic回归模型。

&emsp;&emsp;根据书中第103页6.3节（模型学习的最优化算法）
> &emsp;&emsp;Logistic回归模型、最大熵模型学习归结为以似然函数为目标函数的最优化问题，通常通过迭代算法求解。常用的方法有改进的迭代尺度法、梯度下降法、牛顿法或拟牛顿法。

综上所述：
1. Logistic回归模型的学习策略：极大似然估计法
2. Logistic回归模型的学习算法：改进的迭代尺度法、梯度下降法、牛顿法或拟牛顿法

**第4步：比较支持向量机、 AdaBoost 、Logistic回归模型的学习策略与算法**

| &emsp;&emsp; | 学习策略 | 算法 |
| --- | --- | --- |
| 支持向量机 | 软间隔最大化、最小化由合页损失函数和正则化项组成的目标函数 | 凸二次规划、SMO算法（序列最小最优化算法） |
| AdaBoost | 极小化通过加法模型组成的指数损失函数 | 学习加法模型的前向分步算法 |
| Logistic回归 | 极大似然估计法 | 改进的迭代尺度法、梯度下降法、牛顿法或拟牛顿法 |
