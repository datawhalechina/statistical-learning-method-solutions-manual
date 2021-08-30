# 第11章 条件随机场

## 习题11.1
&emsp;&emsp;写出图11.3中无向图描述的概率图模型的因子分解式。
<br/><center>
<img style="border-radius: 0.3125em;box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="./images/11-1-Maximal-Clique.png"><br><div style="color:orange; border-bottom: 1px solid #d9d9d9;display: inline-block;color: #000;padding: 2px;">图11.3 无向图的团和最大团</div></center>

**解答：**

**解答思路：**

1. 给出无向图中团与最大团的定义；
2. 给出概率无向图模型的因子分解的定义；
3. 计算概率无向图模型的因子分解式。

**解答步骤：**

**第1步：无向图中团与最大团的定义**

&emsp;&emsp;根据书中第217页团与最大团的定义：
> &emsp;&emsp;无向图$G$中任意两个结点均有边连接的结点子集称为团（clique），若$C$是无向图$G$的一个团，并且不能再加进任何一个$G$的结点使其成为一个更大的团，则称此$C$为最大团（maximal clique）。

**第2步：概率无向图模型的因子分解的定义**

&emsp;&emsp;根据书中第218页概率无向图模型的因子分解定义：
> &emsp;&emsp;将概率无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积形式的操作，称为概率无向图模型的因子分解（factorization）。 

&emsp;&emsp;根据书中第218页概率无向图模型的联合概率分布：
> &emsp;&emsp;给定概率无向图模型，设其无向图为$G$，$C$为$G$上的最大团，$Y_C$表示$C$对应的随机变量。那么概率无向图模型的联合概率分布$P(Y)$可写作图中所有最大团$C$上的函数$\Psi_C(Y_C)$的乘积形式，即
> $$
P(Y) = \frac{1}{Z} \prod_C \Psi_C(Y_C)
$$
> 其中，$Z$是规范化因子（normaliztion factor），由式
> $$
Z = \sum_Y \prod_C \Psi_C(Y_C)
$$
> 给出。

**第3步：计算概率无向图模型的因子分解式**

由图11.3可知，该图是由4个结点组成的无向图，结点分别为$Y_1, Y_2, Y_3, Y_4$，根据第1步的团和最大团定义，可得：  
1. 图中由2个结点组成的团有5个：$\{Y_1,Y_2\},\{Y_2,Y_3\},\{Y_3,Y_4\},\{Y_4,Y_2\}$和$\{Y_1,Y_3\}$    
2. 图中包括2个最大团：$\{Y_1,Y_2,Y_3\}$和$\{Y_2,Y_3,Y_4\}$  
3. 由于$Y_1$和$Y_4$没有边连接，所以$\{Y_1,Y_2,Y_3,Y_4\}$不是一个团。  

根据第2步中概率图模型的因子分解定义和联合概率分布的计算公式，可得因子分解：  
$$
P(Y)=\frac{\Psi_{(1,2,3)}(Y_{(1,2,3)})\cdot\Psi_{(2,3,4)}(Y_{(2,3,4)})}
{\displaystyle \sum_Y \left[ \Psi_{(1,2,3)}(Y_{(1,2,3)})\cdot\Psi_{(2,3,4)}(Y_{(2,3,4)})\right]}
$$

## 习题11.2

&emsp;&emsp;证明$Z(x)=a_n^T(x) \cdot \boldsymbol{1} = \boldsymbol{1}^T \cdot \beta_0(x)$，其中$\boldsymbol{1}$是元素均为1的$m$维列向量。

**解答：**

**解答思路：**

1. 给出$Z(x)$的定义公式；
2. 根据书中第225页前向-后向算法，推导$\alpha_n^T(x)$和$\beta_0(x)$；
3. 证明$Z(x) = \alpha_n^T(x) \cdot \boldsymbol{1}$
4. 证明$Z(x) = \boldsymbol{1}^T\cdot\beta_0(x)$

**解答步骤：**

**第1步：给出$Z(x)$的定义式**

&emsp;&emsp;根据书中第223页条件随机场的矩阵形式：
> &emsp;&emsp;假设$P_w(y|x)$是由式(11.15)~式(11.16)给出的线性链条件随机场，表示对给定观测序列$x$，相应的标记序列$y$的条件概率。对每个标记序列引进特殊的起点和终点状态标记$y_0 = \text{start}$和$y_{n+1} = \text{stop}$，这时标注序列的概率$P_w(y|x)$可以通过矩阵形式表示并有效计算。  
> &emsp;&emsp;对观测序列$x$的每一个位置$i=1,2,\cdots, n+1$，由于$y_{i-1}$和$y_i$在$m$个标记中取值，可以定义一个$m$阶矩阵随机变量
> 
> $$
M_i(x) = [M_i(y_{i-1}, y_i | x)]
$$
> 
> 条件概率$P_w(y|x)$是
> 
> $$
P_w(y|x) = \frac{1}{Z_w(x)} \prod_{i=1}^{n+1} M_i(y_{i-1}, y_i | x)
$$
> 
> 其中，$Z_w(x)$为规范化因子，是$n+1$个矩阵的乘积的(start, stop)元素，即
> 
> $$
Z_w(x) = \left[M_1(x)M_2(x)\cdots M_{n+1}(x)\right]_{\text{start}, \text{stop}}
$$
> 
> 注意，$y_0 = \text{start}$与$y_{n+1} = \text{stop}$表示开始状态与终止状态，规范化因子$Z_w(x)$是以start为起点stop为终点，通过状态的所有路径$y_1 y_2 \cdots y_n$的非规范化概率$\displaystyle \prod_{i=1}^{n+1} M_i(y_{i-1}, y_i | x)$之和。

**第2步：给出$\alpha_n^T(x)$和$\beta_1(x)$的定义式**

&emsp;&emsp;根据书中第225页前向-后向算法：
> &emsp;&emsp;对每个指标$i=0,1,\cdots, n+1$，定义前向向量$\alpha_i(x)$：
> $$
\alpha_0(y|x)=\left \{ \begin{array}{ll}
1, & y=\text{start} \\
0, & 否则
\end{array} \right.
$$
> 递推公式为：
> $$ 
\alpha_i^T (y_i|x) = \alpha_{i-1}^T (y_{i-1}|x) M_i(y_{i-1},y_i|x), \quad i=1, 2, \cdots, n+1
$$
> 又可表示为
> $$ 
\alpha_i^T (x) = \alpha_{i-1}^T(x) M_i(x)
$$
> $\alpha_i(y_i|x)$表示在位置$i$的标记是$y_i$，并且从1到$i$的前部分标记序列的非规范化概率，$y_i$可取的值有$m$个，所以$\alpha_i(x)$是$m$维列向量。

&emsp;&emsp;根据书中第225页前向-后向算法：
> &emsp;&emsp;对每个指标$i=0,1,\cdots, n+1$，定义后向向量$\beta_i(x)$：
> 
> $$
\beta_{n+1}(y_{n+1}|x) = \left \{
\begin{array}{ll}
1, & y_{n+1}=\text{stop} \\
0, & 否则
\end{array} \right.
$$
> 
> $$ 
\beta_i(y_i|x) = [M_{i+1}(y_i, y_{i+1} | x)] \beta_{i+1}(y_{i+1} | x)
$$
> 又可表示为
> 
> $$ 
\beta_i(x) = M_{i+1}(x) \beta_{i+1}(x)
$$
> 
> $\beta_i(y_i | x)$表示在位置$i$的标记为$y_i$，并且从$i+1$到$n$的后部分标记序列的非规范化概率。

&emsp;&emsp;根据参考文献 [Shallow Parsing with Conditional Random Fields](http://portal.acm.org/ft_gateway.cfm?id=1073473&type=pdf&CFID=4684435&CFTOKEN=39459323) 中的第2章Conditional Random Fields：
> $$
\alpha_i = \left \{ 
\begin{array}{ll}
\alpha_{i-1} M_i, & 0 < i \leqslant n \\
\boldsymbol{1}, & i = 0
\end{array}
\right.
$$
> 
> $$
\beta_i^T = \left \{ 
\begin{array}{ll}
M_{i+1} \beta_{i+1}^T, & 1 \leqslant i  <  n \\
\boldsymbol{1}, & i = n
\end{array}
\right.
$$

&emsp;&emsp;由上述可得：
1. 当观测序列$x$有n个结点时，有
$$
\alpha_{n}^T(x) = \alpha_0^T(x) M_1(x)\cdots M_{n}(x)
$$

其中$y_0=\text{start}$和$y_{n}=\text{stop}$分别表示开始状态和终止状态，且$Z(x)=[M_1(x) M_2(x) \cdots M_n(x)]_{\text{start, stop}}$


2. 当观测序列$x$有n-1个结点时，有
$$
\beta_1(x) = M_2(x)\cdots M_{n}(x)\beta_{n}(x)
$$

其中$y_1=\text{start}$和$y_{n}=\text{stop}$分别表示开始状态和终止状态，且$Z(x)=[M_2(x) \cdots M_n(x)]_{\text{start, stop}}$

**第3步：证明$Z(x) = \alpha_{n}^T(x) \cdot \boldsymbol{1}$**

$\because \alpha_0(y|x)=\left \{ \begin{array}{ll}
1, & y_0 = \text{start} \\
0, & 否则
\end{array} \right.$  

$\begin{aligned} 
\therefore \alpha_n^T(x) \cdot \boldsymbol{1} 
&= \alpha_0^T(x) M_1(x)\cdots M_{n}(x) \cdot \boldsymbol{1} \\
&= \boldsymbol{1}^T \cdot M_1(x)\cdots M_{n}(x) \cdot \boldsymbol{1} \\
&= Z(x)
\end{aligned}$

**第4步：证明$Z(x)=\boldsymbol{1}^T \cdot \beta_1(x)$**

$\because \beta_{n}(y_{n}|x) = \left \{
\begin{array}{ll}
1, & y_{n}=\text{stop} \\
0, & 否则
\end{array} \right.$

$\begin{aligned}
\therefore \boldsymbol{1}^T \cdot \beta_1(x)
&= \boldsymbol{1}^T \cdot M_2(x) \cdots M_{n}(x) \beta_{n}(x) \\
&= \boldsymbol{1}^T \cdot M_2(x) \cdots M_{n}(x) \cdot \boldsymbol{1} \\
&= Z(x)
\end{aligned}$

综上所述：$Z(x)=a_{n}^T(x) \cdot \boldsymbol{1} = \boldsymbol{1}^T \cdot \beta_1(x)$，命题得证。  

## 习题11.3
&emsp;&emsp;写出条件随机场模型学习的梯度下降法。

**解答：**

**解答思路：**

1. 给出条件随机场模型的对数似然函数；
2. 写出条件随机场模型学习的梯度下降法。

**解答步骤：**

**第1步：条件随机场模型的对数似然函数**

&emsp;&emsp;根据书中第220页线性链条件随机场的参数化形式：
> 设$P(Y|X)$为线性链条件随机场，则在随机变量$X$取值为$x$的条件下，随机变量$Y$取值为$y$的条件概率具有如下形式：
> $$
P(y|x) = \frac{1}{Z(x)} \exp \left( \sum_{i,k} \lambda_k t_k (y_{i-1}, y_i, x, i) + \sum_{i,l} \mu_l s_l (y_i, x, i) \right)
$$
> 其中，
> $$
Z(x) = \sum_y \exp \left( \sum_{i,k} \lambda_k t_k (y_{i-1}, y_i, x, i) + \sum_{i,l} \mu_l s_l (y_i, x, i)  \right)
$$
> 式中，$t_k$和$s_l$是特征函数，$\lambda_k$和$\mu_l$是对应的权值。$Z(x)$是规范化因子，求和是在所有可能的输出序列上进行的。

&emsp;&emsp;根据书中第221页~第222页条件随机场的简化形式：
> 设有$K_1$个转移特征，$K_2$个状态特征，$K=K_1 + K_2$，记
> $$
f_k(y_{i-1}, y_i, x, i) = \left \{ \begin{array}{ll} 
t_k(y_{i-1}, y_i, x, i), & k=1,2,\cdots,K_1 \\
s_l(y_i, x, i), & k=K_1 + l; \quad l = 1,2,\cdots,K_2
\end{array} \right.
$$
> 然后，对转移特征与状态特征在各个位置$i$求和，记作
> $$
f_k(y,x) = \sum_{i=1}^n f_k(y_{i-1}, y_i, x, i), \quad k=1,2, \cdots, K
$$
> 用$w_k$表示特征$f_k(y,x)$的权值，即
> $$
w_k = \left \{ \begin{array}{ll} 
\lambda_k, &  k=1,2,\cdots, K_1 \\
\mu_l, &  k=K_1 + l; \quad l = 1,2,\cdots,K_2
\end{array}\right.
$$
> 于是，条件随机场可表示为
> $$
P(y|x) = \frac{1}{Z(x)} \exp \sum_{k=1}^K w_k f_k(y,x) \\
Z(x) = \sum_y \exp \sum_{k=1}^K w_k f_k(y,x)
$$

&emsp;&emsp;根据书中第227页条件随机场模型的对数似然函数：
> 当$P_w$是一个由式(11.15)和(11.16)给出的条件随机场模型时，对数似然函数为
> $$
L(w)=\sum^N_{j=1} \sum^K_{k=1} w_k f_k(y_j,x_j)-\sum^N_{j=1} \log{Z_w(x_j)}
$$

&emsp;&emsp;将对数似然函数求偏导，可得
$$
\begin{aligned}
g(w^{(n)}) = \frac{\partial L(w)}{\partial w^{(n)}}
&= \sum^N_{j=1} f_n(y_j,x_j) - \sum^N_{j=1} \frac{1}{Z_w(x_j)} \cdot \frac{\partial{Z_w(x_j)}}{\partial{w_n}}\\
&= \sum^N_{j=1} f_n(y_j,x_j) - \sum^N_{i=1} \frac{1}{Z_w(x_i)} \cdot \sum^N_{j=1} \left[ \left( \exp{\sum^K_{k=1} w_k f_k (y_j,x_i)} \right) \cdot f_n(y_j, x_i) \right]
\end{aligned}
$$

&emsp;&emsp;梯度函数为
$$
\nabla L(w)=\left[\frac{\partial L(w)}{\partial w^{(0)}}, \cdots, \frac{\partial L(w)}{\partial w^{(N)}}\right]
$$

**第2步：写出条件随机场模型学习的梯度下降法**

&emsp;&emsp;根据书中第439页附录A 梯度下降法的算法：
> 输入：目标函数$f(x)$，梯度函数$g(x)= \nabla f(x)$，计算精度$\varepsilon$；  
输出：$f(x)$的极小值$x^*$。  
(1) 取初始值$x^{(0)} \in R^n$，置$k=0$  
(2) 计算$f(x^{(k)})$  
(3) 计算梯度$g_k=g(x^{(k)})$，当$\|g_k\| < \varepsilon$时，停止迭代，令$x^* = x^{(k)}$；否则，令$p_k=-g(x^{(k)})$，求$\lambda_k$，使
> 
> $$
\displaystyle f(x^{(k)}+\lambda_k p_k) = \min \limits_{\lambda \geqslant 0}f(x^{(k)}+\lambda p_k)
$$
> (4) 置$x^{(k+1)}=x^{(k)}+\lambda_k p_k$，计算$f(x^{(k+1)})$  
当$\|f(x^{(k+1)}) - f(x^{(k)})\| < \varepsilon$或 $\|x^{(k+1)} - x^{(k)}\| < \varepsilon$时，停止迭代，令$x^* = x^{(k+1)}$  
(5) 否则，置$k=k+1$，转(3)

条件随机场模型学习的梯度下降法：  

输入：目标函数$f(w)$，梯度函数$g(w) = \nabla f(w)$，计算精度$\varepsilon$  
输出：$f(w)$的极大值$w^*$  
(1) 取初始值$w^{(0)} \in R^n$，置$k=0$  
(2) 计算$f(w^{(n)})$  
(3) 计算梯度$g_n=g(w^{(n)})$，当$\|g_n\| < \varepsilon$时，停止迭代，令$w^* = w^{(n)}$；否则，令$p_n=-g(w^{(n)})$，求$\lambda_n$，使

$$
\displaystyle f(w^{(n)}+\lambda_n p_n) = \max_{\lambda \geqslant 0}f(w^{(n)}+\lambda p_n)
$$  
(4) 置$w^{(n+1)}=w^{(n)}+\lambda_n p_n$，计算$f(w^{(n+1)})$  
当$\|f(w^{(n+1)}) - f(w^{(n)})\| < \varepsilon$或 $\|w^{(n+1)} - w^{(n)}\| < \varepsilon$时，停止迭代，令$w^* = w^{(n+1)}$  
(5) 否则，置$n=n+1$，转(3)

## 习题11.4

参考图11.6的状态路径图，假设随机矩阵$M_1(x),M_2(x),M_3(x),M_4(x)$分别是
$$
M_1(x)=\begin{bmatrix}0&0 \\ 0.5&0.5 \end{bmatrix} ,
M_2(x)=\begin{bmatrix}0.3&0.7 \\ 0.7&0.3\end{bmatrix} \\
M_3(x)=\begin{bmatrix}0.5&0.5 \\ 0.6&0.4\end{bmatrix},
M_4(x)=\begin{bmatrix}0&1 \\ 0&1\end{bmatrix}
$$
求以$\text{start}=2$为起点$\text{stop}=2$为终点的所有路径的状态序列$y$的概率及概率最大的状态序列。

**解答：**

**解答思路：**  

&emsp;&emsp;根据书中第223页条件随机场的矩阵形式，以及例题11.2，通过自编程实现计算所有路径状态序列的概率及概率最大的状态序列。

**解答步骤：**


```python
import numpy as np


class CRFMatrix:
    def __init__(self, M, start, stop):
        # 随机矩阵
        self.M = M
        #
        self.start = start
        self.stop = stop
        self.path_prob = None

    def _create_path(self):
        """按照图11.6的状态路径图，生成路径"""
        # 初始化start结点
        path = [self.start]
        for i in range(1, len(self.M)):
            paths = []
            for _, r in enumerate(path):
                temp = np.transpose(r)
                # 添加状态结点1
                paths.append(np.append(temp, 1))
                # 添加状态结点2
                paths.append(np.append(temp, 2))
            path = paths.copy()

        # 添加stop结点
        path = [np.append(r, self.stop) for _, r in enumerate(path)]
        return path

    def fit(self):
        path = self._create_path()
        pr = []
        for _, row in enumerate(path):
            p = 1
            for i in range(len(row) - 1):
                a = row[i]
                b = row[i + 1]
                # 根据公式11.24，计算条件概率
                p *= M[i][a - 1][b - 1]
            pr.append((row.tolist(), p))
        # 按照概率从大到小排列
        pr = sorted(pr, key=lambda x: x[1], reverse=True)
        self.path_prob = pr

    def print(self):
        # 打印结果
        print("以start=%s为起点stop=%s为终点的所有路径的状态序列y的概率为：" % (self.start, self.stop))
        for path, p in self.path_prob:
            print("    路径为：" + "->".join([str(x) for x in path]), end=" ")
            print("概率为：" + str(p))
        print("概率最大[" + str(self.path_prob[0][1]) + "]的状态序列为:",
              "->".join([str(x) for x in self.path_prob[0][0]]))
```


```python
# 创建随机矩阵
M1 = [[0, 0], [0.5, 0.5]]
M2 = [[0.3, 0.7], [0.7, 0.3]]
M3 = [[0.5, 0.5], [0.6, 0.4]]
M4 = [[0, 1], [0, 1]]
M = [M1, M2, M3, M4]
# 构建条件随机场的矩阵模型
crf = CRFMatrix(M=M, start=2, stop=2)
# 得到所有路径的状态序列的概率
crf.fit()
# 打印结果
crf.print()
```

    以start=2为起点stop=2为终点的所有路径的状态序列y的概率为：
        路径为：2->1->2->1->2 概率为：0.21
        路径为：2->2->1->1->2 概率为：0.175
        路径为：2->2->1->2->2 概率为：0.175
        路径为：2->1->2->2->2 概率为：0.13999999999999999
        路径为：2->2->2->1->2 概率为：0.09
        路径为：2->1->1->1->2 概率为：0.075
        路径为：2->1->1->2->2 概率为：0.075
        路径为：2->2->2->2->2 概率为：0.06
    概率最大[0.21]的状态序列为: 2->1->2->1->2
    
