# 第10章 隐马尔可夫模型

## 习题10.1
&emsp;&emsp;给定盒子和球组成的隐马尔可夫模型$\lambda=(A,B,\pi)$，其中，

$$
A=\left[\begin{array}{ccc}
0.5&0.2&0.3 \\
0.3&0.5&0.2 \\
0.2&0.3&0.5
\end{array}\right], 
\quad B=\left[\begin{array}{cc}
0.5&0.5 \\
0.4&0.6 \\
0.7&0.3
\end{array}\right], 
\quad \pi=(0.2,0.4,0.4)^T
$$

设$T=4$，$O=(红,白,红,白)$，试用后向算法计算$P(O|\lambda)$。

**解答：**

**解答思路：**
1. 列出隐马尔可夫模型的定义
2. 列出后向算法
3. 自编码实现隐马尔可夫的后向算法

**解答步骤：**

**第1步：隐马尔可夫模型**

&emsp;&emsp;根据书中第193页隐马尔可夫模型的定义：
> 1. 条件假设：  
> &emsp;&emsp;设$Q$是所有可能的状态的集合，$V$是所有可能的观测的集合：
> $$
Q=\{q_1, q_2, \cdots, q_N\}, \quad V=\{v_1, v_2, \cdots, v_M\}
$$
> 其中，$N$是可能的状态数，$M$是可能的观测数。  
> &emsp;&emsp;$I$是长度为$T$的状态序列，$O$是对应的观测序列：
> $$
I = (i_1,i_2, \cdots, i_T), \quad O=(o_1, o_2, \cdots, o_T)
$$  
> 
> 2. 状态转移概率矩阵$A$：  
> &emsp;&emsp;$A$是状态转移概率矩阵：
> $$
A = [a_{ij}]_{N \times N}
$$  
> 其中，
> $$
a_{ij} = P(i_{t+1}=q_j | i_t = q_i), \quad i=1,2,\cdots, N; \quad j=1,2,\cdots N
$$
> 是在时刻$t$处于状态$q_i$的条件下，在时刻$t+1$转移到状态$q_j$的概率。  
>   
> 3. 观测概率矩阵$B$：  
> &emsp;&emsp;$B$是观测概率矩阵：
> $$
B = [b_j(k)]_{N \times M}
$$  
> 其中，
> $$
b_j(k) = P(o_t=v_k | i_t = q_j), \quad k=1,2,\cdots, M; \quad j=1,2,\cdots N
$$
> 是在时刻$t$处于状态$q_j$的条件下，生成观测$v_k$的概率。
>
> 4. 初始状态概率向量$\pi$：  
> &emsp;&emsp;$\pi$是初始状态概率向量：
> $$
\pi = (\pi_i)
$$  
> 其中，
> $$
\pi = P(i_1 = q_i), \quad i=1,2,\cdots N
$$
> 是时刻$t=1$处于状态$q_i$的概率。
> 
> 5. 隐马尔可夫模型的表示：  
> &emsp;&emsp;隐马尔可夫模型由初始状态概率向量$\pi$、状态转移概率矩阵$A$和观测概率矩阵$B$决定。$\pi$和$A$决定状态序列，$B$决定观测序列。因此隐马尔可夫模型$\lambda$可以用三元符号表示，即
> $$
\lambda = (A, B, \pi)
$$
> $A,B,\pi$称为隐马尔可夫模型的三要素。

**第2步：后向算法**

&emsp;&emsp;根据书中第201页的观测序列概率的后向算法：
> 输入：隐马尔可夫模型$\lambda$，观测序列$O$  
输出：观测序列概率$P(O|\lambda)$  
（1）
> $$
\beta_T(i) = 1, \quad i = 1, 2, \cdots, N
$$  
>（2）对$t= T-1, T-2, \cdots, 1$
> $$
\beta_t(i) = \sum_{j=1}^N a_{ij} b_j(o_{t+1}) \beta_{t+1}(j), \quad  i=1,2,\cdots, N
$$
>（3）
> $$
P(O|\lambda) = \sum_{i=1}^N \pi_i b_i(o_1) \beta_1(i)
$$  

**第3步：自编码实现隐马尔可夫的后向算法**


```python
import numpy as np


class HiddenMarkovBackward:
    def __init__(self):
        self.betas = None
        self.backward_P = None

    def backward(self, Q, V, A, B, O, PI):
        """
        后向算法
        :param Q: 所有可能的状态集合
        :param V: 所有可能的观测集合
        :param A: 状态转移概率矩阵
        :param B: 观测概率矩阵
        :param O: 观测序列
        :param PI: 初始状态概率向量
        """
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # (1)初始化后向概率beta值，书中第201页公式(10.19)
        betas = np.ones((N, M))
        self.print_betas_T(N, M)

        # (2)对观测序列逆向遍历，M-2即为T-1
        print("\n从时刻T-1到1观测序列的后向概率：")
        for t in range(M - 2, -1, -1):
            # 得到序列对应的索引
            index_of_o = V.index(O[t + 1])
            # 遍历状态序列
            for i in range(N):
                # 书中第201页公式(10.20)
                betas[i][t] = np.dot(np.multiply(A[i], [b[index_of_o] for b in B]),
                                     [beta[t + 1] for beta in betas])
                real_t = t + 1
                real_i = i + 1
                self.print_betas_t(A, B, N, betas, i, index_of_o, real_i, real_t, t)

        # 取出第一个值索引，用于得到o1
        index_of_o = V.index(O[0])
        self.betas = betas
        # 书中第201页公式(10.21)
        P = np.dot(np.multiply(PI, [b[index_of_o] for b in B]),
                   [beta[0] for beta in betas])
        self.backward_P = P
        self.print_P(B, N, P, PI, betas, index_of_o)

    @staticmethod
    def print_P(B, N, P, PI, betas, index_of_o):
        print("\n观测序列概率：")
        print("P(O|lambda) = ", end="")
        for i in range(N):
            print("%.1f * %.1f * %.5f + "
                  % (PI[0][i], B[i][index_of_o], betas[i][0]), end="")
        print("0 = %f" % P)

    @staticmethod
    def print_betas_t(A, B, N, betas, i, index_of_o, real_i, real_t, t):
        print("beta%d(%d) = sum[a%dj * bj(o%d) * beta%d(j)] = ("
              % (real_t, real_i, real_i, real_t + 1, real_t + 1), end='')
        for j in range(N):
            print("%.2f * %.2f * %.2f + "
                  % (A[i][j], B[j][index_of_o], betas[j][t + 1]), end='')
        print("0) = %.3f" % betas[i][t])

    @staticmethod
    def print_betas_T(N, M):
        print("初始化后向概率：")
        for i in range(N):
            print('beta%d(%d) = 1' % (M, i + 1))
```


```python
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '白']
PI = [[0.2, 0.4, 0.4]]

hmm_backward = HiddenMarkovBackward()
hmm_backward.backward(Q, V, A, B, O, PI)
```

    初始化后向概率：
    beta4(1) = 1
    beta4(2) = 1
    beta4(3) = 1
    
    从时刻T-1到1观测序列的后向概率：
    beta3(1) = sum[a1j * bj(o4) * beta4(j)] = (0.50 * 0.50 * 1.00 + 0.20 * 0.60 * 1.00 + 0.30 * 0.30 * 1.00 + 0) = 0.460
    beta3(2) = sum[a2j * bj(o4) * beta4(j)] = (0.30 * 0.50 * 1.00 + 0.50 * 0.60 * 1.00 + 0.20 * 0.30 * 1.00 + 0) = 0.510
    beta3(3) = sum[a3j * bj(o4) * beta4(j)] = (0.20 * 0.50 * 1.00 + 0.30 * 0.60 * 1.00 + 0.50 * 0.30 * 1.00 + 0) = 0.430
    beta2(1) = sum[a1j * bj(o3) * beta3(j)] = (0.50 * 0.50 * 0.46 + 0.20 * 0.40 * 0.51 + 0.30 * 0.70 * 0.43 + 0) = 0.246
    beta2(2) = sum[a2j * bj(o3) * beta3(j)] = (0.30 * 0.50 * 0.46 + 0.50 * 0.40 * 0.51 + 0.20 * 0.70 * 0.43 + 0) = 0.231
    beta2(3) = sum[a3j * bj(o3) * beta3(j)] = (0.20 * 0.50 * 0.46 + 0.30 * 0.40 * 0.51 + 0.50 * 0.70 * 0.43 + 0) = 0.258
    beta1(1) = sum[a1j * bj(o2) * beta2(j)] = (0.50 * 0.50 * 0.25 + 0.20 * 0.60 * 0.23 + 0.30 * 0.30 * 0.26 + 0) = 0.112
    beta1(2) = sum[a2j * bj(o2) * beta2(j)] = (0.30 * 0.50 * 0.25 + 0.50 * 0.60 * 0.23 + 0.20 * 0.30 * 0.26 + 0) = 0.122
    beta1(3) = sum[a3j * bj(o2) * beta2(j)] = (0.20 * 0.50 * 0.25 + 0.30 * 0.60 * 0.23 + 0.50 * 0.30 * 0.26 + 0) = 0.105
    
    观测序列概率：
    P(O|lambda) = 0.2 * 0.5 * 0.11246 + 0.4 * 0.4 * 0.12174 + 0.4 * 0.7 * 0.10488 + 0 = 0.060091
    

可得$P(O|\lambda) = 0.060091$

## 习题10.2

&emsp;&emsp;给定盒子和球组成的隐马尔可夫模型$\lambda=(A,B,\pi)$，其中，

$$
A=\left[
\begin{array}{ccc}
0.5&0.1&0.4 \\
0.3&0.5&0.2 \\
0.2&0.2&0.6 
\end{array}\right], 
\quad B=\left[
\begin{array}{cc}
0.5&0.5 \\
0.4&0.6 \\
0.7&0.3
\end{array}\right], 
\quad \pi=(0.2,0.3,0.5)^T
$$

设$T=8$，$O=(红,白,红,红,白,红,白,白)$，试用前向后向概率计算$P(i_4=q_3|O,\lambda)$

**解答：**

**解答思路：**
1. 列出前向算法
2. 根据前向概率和后向概率，列出单个状态概率的计算公式
2. 自编程实现用前向后向概率计算$P(i_4=q_3|O,\lambda)$

**解答步骤：**

**第1步：前向算法**

&emsp;&emsp;根据书中第198页观测序列概率的前向算法：
> 输入：隐马尔可夫模型$\lambda$，观测序列$O$；  
输出：观测序列概率$P(O|\lambda)$。  
（1）初值
> $$
\alpha_1(i) = \pi_i b_i(o_1), \quad i=1,2,\cdots, N
$$  
>（2）递推，对$t=1,2,\cdots,T-1，$
> $$
\alpha_{t+1}(i) = \left[ \sum_{j=1}^N \alpha_t(j) a_{ji} \right] b_i(o_{t+1})， \quad i=1,2,\cdots, N
$$
>（3）终止
> $$
P(O|\lambda) = \sum_{i=1}^N \alpha_T(i)
$$

**第2步：单个状态概率的计算公式**

&emsp;&emsp;根据书中第202页单个状态概率的计算公式：
> &emsp;&emsp;利用前向概率和后向概率，给定模型$\lambda$和观测$O$，在时刻$t$处于状态$q_i$的概率
> $$
\begin{aligned}
\gamma_t(i) &= P(i_t = q_i |O, \lambda) \\
&= \frac{P(i_t=q_i,O|\lambda)}{P(O|\lambda)} \\
&= \frac{\alpha_t(i) \beta_t(i)}{P(O|\lambda)}
\end{aligned}
$$

**第3步：自编程实现前向后向算法**


```python
import numpy as np


class HiddenMarkovForwardBackward(HiddenMarkovBackward):
    def __init__(self, verbose=False):
        super(HiddenMarkovBackward, self).__init__()
        self.alphas = None
        self.forward_P = None
        self.verbose = verbose

    def forward(self, Q, V, A, B, O, PI):
        """
        前向算法
        :param Q: 所有可能的状态集合
        :param V: 所有可能的观测集合
        :param A: 状态转移概率矩阵
        :param B: 观测概率矩阵
        :param O: 观测序列
        :param PI: 初始状态概率向量
        """
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # 初始化前向概率alpha值
        alphas = np.zeros((N, M))
        # 时刻数=观测序列数
        T = M
        # (2)对观测序列遍历，遍历每一个时刻，计算前向概率alpha值

        for t in range(T):
            if self.verbose:
                if t == 0:
                    print("前向概率初值：")
                elif t == 1:
                    print("\n从时刻1到T-1观测序列的前向概率：")
            # 得到序列对应的索引
            index_of_o = V.index(O[t])
            # 遍历状态序列
            for i in range(N):
                if t == 0:
                    # (1)初始化alpha初值，书中第198页公式(10.15)
                    alphas[i][t] = PI[t][i] * B[i][index_of_o]
                    if self.verbose:
                        self.print_alpha_t1(alphas, i, t)
                else:
                    # (2)递推，书中第198页公式(10.16)
                    alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas],
                                          [a[i] for a in A]) * B[i][index_of_o]
                    if self.verbose:
                        self.print_alpha_t(alphas, i, t)
        # (3)终止，书中第198页公式(10.17)
        self.forward_P = np.sum([alpha[M - 1] for alpha in alphas])
        self.alphas = alphas

    @staticmethod
    def print_alpha_t(alphas, i, t):
        print("alpha%d(%d) = [sum alpha%d(i) * ai%d] * b%d(o%d) = %f"
              % (t + 1, i + 1, t - 1, i, i, t, alphas[i][t]))

    @staticmethod
    def print_alpha_t1(alphas, i, t):
        print('alpha1(%d) = pi%d * b%d * b(o1) = %f'
              % (i + 1, i, i, alphas[i][t]))

    def calc_t_qi_prob(self, t, qi):
        result = (self.alphas[qi - 1][t - 1] * self.betas[qi - 1][t - 1]) / self.backward_P[0]
        if self.verbose:
            print("计算P(i%d=q%d|O,lambda)：" % (t, qi))
            print("P(i%d=q%d|O,lambda) = alpha%d(%d) * beta%d(%d) / P(O|lambda) = %f"
                  % (t, qi, t, qi, t, qi, result))

        return result
```


```python
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '红', '白', '红', '白', '白']
PI = [[0.2, 0.3, 0.5]]

hmm_forward_backward = HiddenMarkovForwardBackward(verbose=True)
hmm_forward_backward.forward(Q, V, A, B, O, PI)
print()
hmm_forward_backward.backward(Q, V, A, B, O, PI)
print()
hmm_forward_backward.calc_t_qi_prob(t=4, qi=3)
print()
```

    前向概率初值：
    alpha1(1) = pi0 * b0 * b(o1) = 0.100000
    alpha1(2) = pi1 * b1 * b(o1) = 0.120000
    alpha1(3) = pi2 * b2 * b(o1) = 0.350000
    
    从时刻1到T-1观测序列的前向概率：
    alpha2(1) = [sum alpha0(i) * ai0] * b0(o1) = 0.078000
    alpha2(2) = [sum alpha0(i) * ai1] * b1(o1) = 0.084000
    alpha2(3) = [sum alpha0(i) * ai2] * b2(o1) = 0.082200
    alpha3(1) = [sum alpha1(i) * ai0] * b0(o2) = 0.040320
    alpha3(2) = [sum alpha1(i) * ai1] * b1(o2) = 0.026496
    alpha3(3) = [sum alpha1(i) * ai2] * b2(o2) = 0.068124
    alpha4(1) = [sum alpha2(i) * ai0] * b0(o3) = 0.020867
    alpha4(2) = [sum alpha2(i) * ai1] * b1(o3) = 0.012362
    alpha4(3) = [sum alpha2(i) * ai2] * b2(o3) = 0.043611
    alpha5(1) = [sum alpha3(i) * ai0] * b0(o4) = 0.011432
    alpha5(2) = [sum alpha3(i) * ai1] * b1(o4) = 0.010194
    alpha5(3) = [sum alpha3(i) * ai2] * b2(o4) = 0.011096
    alpha6(1) = [sum alpha4(i) * ai0] * b0(o5) = 0.005497
    alpha6(2) = [sum alpha4(i) * ai1] * b1(o5) = 0.003384
    alpha6(3) = [sum alpha4(i) * ai2] * b2(o5) = 0.009288
    alpha7(1) = [sum alpha5(i) * ai0] * b0(o6) = 0.002811
    alpha7(2) = [sum alpha5(i) * ai1] * b1(o6) = 0.002460
    alpha7(3) = [sum alpha5(i) * ai2] * b2(o6) = 0.002535
    alpha8(1) = [sum alpha6(i) * ai0] * b0(o7) = 0.001325
    alpha8(2) = [sum alpha6(i) * ai1] * b1(o7) = 0.001211
    alpha8(3) = [sum alpha6(i) * ai2] * b2(o7) = 0.000941
    
    初始化后向概率：
    beta8(1) = 1
    beta8(2) = 1
    beta8(3) = 1
    
    从时刻T-1到1观测序列的后向概率：
    beta7(1) = sum[a1j * bj(o8) * beta8(j)] = (0.50 * 0.50 * 1.00 + 0.10 * 0.60 * 1.00 + 0.40 * 0.30 * 1.00 + 0) = 0.430
    beta7(2) = sum[a2j * bj(o8) * beta8(j)] = (0.30 * 0.50 * 1.00 + 0.50 * 0.60 * 1.00 + 0.20 * 0.30 * 1.00 + 0) = 0.510
    beta7(3) = sum[a3j * bj(o8) * beta8(j)] = (0.20 * 0.50 * 1.00 + 0.20 * 0.60 * 1.00 + 0.60 * 0.30 * 1.00 + 0) = 0.400
    beta6(1) = sum[a1j * bj(o7) * beta7(j)] = (0.50 * 0.50 * 0.43 + 0.10 * 0.60 * 0.51 + 0.40 * 0.30 * 0.40 + 0) = 0.186
    beta6(2) = sum[a2j * bj(o7) * beta7(j)] = (0.30 * 0.50 * 0.43 + 0.50 * 0.60 * 0.51 + 0.20 * 0.30 * 0.40 + 0) = 0.241
    beta6(3) = sum[a3j * bj(o7) * beta7(j)] = (0.20 * 0.50 * 0.43 + 0.20 * 0.60 * 0.51 + 0.60 * 0.30 * 0.40 + 0) = 0.176
    beta5(1) = sum[a1j * bj(o6) * beta6(j)] = (0.50 * 0.50 * 0.19 + 0.10 * 0.40 * 0.24 + 0.40 * 0.70 * 0.18 + 0) = 0.106
    beta5(2) = sum[a2j * bj(o6) * beta6(j)] = (0.30 * 0.50 * 0.19 + 0.50 * 0.40 * 0.24 + 0.20 * 0.70 * 0.18 + 0) = 0.101
    beta5(3) = sum[a3j * bj(o6) * beta6(j)] = (0.20 * 0.50 * 0.19 + 0.20 * 0.40 * 0.24 + 0.60 * 0.70 * 0.18 + 0) = 0.112
    beta4(1) = sum[a1j * bj(o5) * beta5(j)] = (0.50 * 0.50 * 0.11 + 0.10 * 0.60 * 0.10 + 0.40 * 0.30 * 0.11 + 0) = 0.046
    beta4(2) = sum[a2j * bj(o5) * beta5(j)] = (0.30 * 0.50 * 0.11 + 0.50 * 0.60 * 0.10 + 0.20 * 0.30 * 0.11 + 0) = 0.053
    beta4(3) = sum[a3j * bj(o5) * beta5(j)] = (0.20 * 0.50 * 0.11 + 0.20 * 0.60 * 0.10 + 0.60 * 0.30 * 0.11 + 0) = 0.043
    beta3(1) = sum[a1j * bj(o4) * beta4(j)] = (0.50 * 0.50 * 0.05 + 0.10 * 0.40 * 0.05 + 0.40 * 0.70 * 0.04 + 0) = 0.026
    beta3(2) = sum[a2j * bj(o4) * beta4(j)] = (0.30 * 0.50 * 0.05 + 0.50 * 0.40 * 0.05 + 0.20 * 0.70 * 0.04 + 0) = 0.023
    beta3(3) = sum[a3j * bj(o4) * beta4(j)] = (0.20 * 0.50 * 0.05 + 0.20 * 0.40 * 0.05 + 0.60 * 0.70 * 0.04 + 0) = 0.027
    beta2(1) = sum[a1j * bj(o3) * beta3(j)] = (0.50 * 0.50 * 0.03 + 0.10 * 0.40 * 0.02 + 0.40 * 0.70 * 0.03 + 0) = 0.015
    beta2(2) = sum[a2j * bj(o3) * beta3(j)] = (0.30 * 0.50 * 0.03 + 0.50 * 0.40 * 0.02 + 0.20 * 0.70 * 0.03 + 0) = 0.012
    beta2(3) = sum[a3j * bj(o3) * beta3(j)] = (0.20 * 0.50 * 0.03 + 0.20 * 0.40 * 0.02 + 0.60 * 0.70 * 0.03 + 0) = 0.016
    beta1(1) = sum[a1j * bj(o2) * beta2(j)] = (0.50 * 0.50 * 0.01 + 0.10 * 0.60 * 0.01 + 0.40 * 0.30 * 0.02 + 0) = 0.006
    beta1(2) = sum[a2j * bj(o2) * beta2(j)] = (0.30 * 0.50 * 0.01 + 0.50 * 0.60 * 0.01 + 0.20 * 0.30 * 0.02 + 0) = 0.007
    beta1(3) = sum[a3j * bj(o2) * beta2(j)] = (0.20 * 0.50 * 0.01 + 0.20 * 0.60 * 0.01 + 0.60 * 0.30 * 0.02 + 0) = 0.006
    
    观测序列概率：
    P(O|lambda) = 0.2 * 0.5 * 0.00633 + 0.3 * 0.4 * 0.00685 + 0.5 * 0.7 * 0.00578 + 0 = 0.003477
    
    计算P(i4=q3|O,lambda)：
    P(i4=q3|O,lambda) = alpha4(3) * beta4(3) / P(O|lambda) = 0.536952
    
    

可知，$\displaystyle P(i_4=q_3|O,\lambda)=\frac{P(i_4=q_3,O|\lambda)}{P(O|\lambda)}=\frac{\alpha_4(3)\beta_4(3)}{P(O|\lambda)} = 0.536952$

## 习题10.3

&emsp;&emsp;在习题10.1中，试用维特比算法求最优路径$I^*=(i_1^*,i_2^*,i_3^*,i_4^*)$。

**解答：**

**解答思路：**
1. 列出维特比算法
2. 自编程实现维特比算法，并求最优路径

**解答步骤：**

**第1步：维特比算法**

&emsp;&emsp;根据书中第209页维特比算法：
> 输入：模型$\lambda=(A,B, \pi)$和观测$O=(o_1, o_2, \cdots, o_T)$；  
输出：最优路径$I^* = (i_1^*, i_2^*, \cdots, i_T^*)$。  
（1）初始化
> $$
\delta_1(i) = \pi_i b_i(o_1), \quad i=1,2, \cdots, N \\
\Psi_1(i) = 0, \quad i=1,2, \cdots, N
$$  
>（2）递推。对$t=2,3, \cdots, T$
> $$
\delta_t(i) = \max \limits_{1 \leqslant j \leqslant N} [\delta_{t-1}(j) a_{ji}] b_i(o_t), \quad i=1,2, \cdots, N \\
\Psi_t(i) = \arg \max \limits_{1 \leqslant j \leqslant N} [\delta_{t-1}(j) a_{ji}], \quad i=1,2, \cdots, N
$$
>（3）终止
> $$
P^* = \max \limits_{1 \leqslant i \leqslant N} \delta_T(i) \\
i_T^* = \arg \max \limits_{1 \leqslant i \leqslant N} [\delta_T(i)]
$$
>（4）最优路径回溯。对$t=T-1, T-2, \cdots , 1$
> $$
i_t^* = \Psi_{t+1}(i_{t+1}^*)
$$
> 求得最优路径$I^* = (i_1^*, i_2^*, \cdots, i_T^*)$。

**第2步：自编程实现维特比算法**


```python
import numpy as np


class HiddenMarkovViterbi:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def viterbi(self, Q, V, A, B, O, PI):
        """
        维特比算法
        :param Q: 所有可能的状态集合
        :param V: 所有可能的观测集合
        :param A: 状态转移概率矩阵
        :param B: 观测概率矩阵
        :param O: 观测序列
        :param PI: 初始状态概率向量
        """
        # 状态序列的大小
        N = len(Q)
        # 观测序列的大小
        M = len(O)
        # 初始化deltas
        deltas = np.zeros((N, M))
        # 初始化psis
        psis = np.zeros((N, M))

        # 初始化最优路径矩阵，该矩阵维度与观测序列维度相同
        I = np.zeros((1, M))
        # (2)递推，遍历观测序列
        for t in range(M):
            if self.verbose:
                if t == 0:
                    print("初始化Psi1和delta1：")
                elif t == 1:
                    print("\n从时刻2到T的所有单个路径中概率"
                          "最大值delta和概率最大的路径的第t-1个结点Psi：")

            # (2)递推从t=2开始
            real_t = t + 1
            # 得到序列对应的索引
            index_of_o = V.index(O[t])
            for i in range(N):
                real_i = i + 1
                if t == 0:
                    # (1)初始化
                    deltas[i][t] = PI[0][i] * B[i][index_of_o]
                    psis[i][t] = 0

                    self.print_delta_t1(
                        B, PI, deltas, i, index_of_o, real_i, t)
                    self.print_psi_t1(real_i)
                else:
                    # (2)递推，对t=2,3,...,T
                    deltas[i][t] = np.max(np.multiply([delta[t - 1] for delta in deltas],
                                                      [a[i] for a in A])) * B[i][index_of_o]
                    self.print_delta_t(
                        A, B, deltas, i, index_of_o, real_i, real_t, t)

                    psis[i][t] = np.argmax(np.multiply([delta[t - 1] for delta in deltas],
                                                       [a[i] for a in A]))
                    self.print_psi_t(i, psis, real_i, real_t, t)

        last_deltas = [delta[M - 1] for delta in deltas]
        # (3)终止，得到所有路径的终结点最大的概率值
        P = np.max(last_deltas)
        # (3)得到最优路径的终结点
        I[0][M - 1] = np.argmax(last_deltas)
        if self.verbose:
            print("\n所有路径的终结点最大的概率值：")
            print("P = %f" % P)
        if self.verbose:
            print("\n最优路径的终结点：")
            print("i%d = argmax[deltaT(i)] = %d" % (M, I[0][M - 1] + 1))
            print("\n最优路径的其他结点：")

        # (4)递归由后向前得到其他结点
        for t in range(M - 2, -1, -1):
            I[0][t] = psis[int(I[0][t + 1])][t + 1]
            if self.verbose:
                print("i%d = Psi%d(i%d) = %d" %
                      (t + 1, t + 2, t + 2, I[0][t] + 1))

        # 输出最优路径
        print("\n最优路径是：", "->".join([str(int(i + 1)) for i in I[0]]))

    def print_psi_t(self, i, psis, real_i, real_t, t):
        if self.verbose:
            print("Psi%d(%d) = argmax[delta%d(j) * aj%d] = %d"
                  % (real_t, real_i, real_t - 1, real_i, psis[i][t]))

    def print_delta_t(self, A, B, deltas, i, index_of_o, real_i, real_t, t):
        if self.verbose:
            print("delta%d(%d) = max[delta%d(j) * aj%d] * b%d(o%d) = %.2f * %.2f = %.5f"
                  % (real_t, real_i, real_t - 1, real_i, real_i, real_t,
                     np.max(np.multiply([delta[t - 1] for delta in deltas],
                                        [a[i] for a in A])),
                     B[i][index_of_o], deltas[i][t]))

    def print_psi_t1(self, real_i):
        if self.verbose:
            print("Psi1(%d) = 0" % real_i)

    def print_delta_t1(self, B, PI, deltas, i, index_of_o, real_i, t):
        if self.verbose:
            print("delta1(%d) = pi%d * b%d(o1) = %.2f * %.2f = %.2f"
                  % (real_i, real_i, real_i, PI[0][i], B[i][index_of_o], deltas[i][t]))
```


```python
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '白']
PI = [[0.2, 0.4, 0.4]]

HMM = HiddenMarkovViterbi(verbose=True)
HMM.viterbi(Q, V, A, B, O, PI)
```

    初始化Psi1和delta1：
    delta1(1) = pi1 * b1(o1) = 0.20 * 0.50 = 0.10
    Psi1(1) = 0
    delta1(2) = pi2 * b2(o1) = 0.40 * 0.40 = 0.16
    Psi1(2) = 0
    delta1(3) = pi3 * b3(o1) = 0.40 * 0.70 = 0.28
    Psi1(3) = 0
    
    从时刻2到T的所有单个路径中概率最大值delta和概率最大的路径的第t-1个结点Psi：
    delta2(1) = max[delta1(j) * aj1] * b1(o2) = 0.06 * 0.50 = 0.02800
    Psi2(1) = argmax[delta1(j) * aj1] = 2
    delta2(2) = max[delta1(j) * aj2] * b2(o2) = 0.08 * 0.60 = 0.05040
    Psi2(2) = argmax[delta1(j) * aj2] = 2
    delta2(3) = max[delta1(j) * aj3] * b3(o2) = 0.14 * 0.30 = 0.04200
    Psi2(3) = argmax[delta1(j) * aj3] = 2
    delta3(1) = max[delta2(j) * aj1] * b1(o3) = 0.02 * 0.50 = 0.00756
    Psi3(1) = argmax[delta2(j) * aj1] = 1
    delta3(2) = max[delta2(j) * aj2] * b2(o3) = 0.03 * 0.40 = 0.01008
    Psi3(2) = argmax[delta2(j) * aj2] = 1
    delta3(3) = max[delta2(j) * aj3] * b3(o3) = 0.02 * 0.70 = 0.01470
    Psi3(3) = argmax[delta2(j) * aj3] = 2
    delta4(1) = max[delta3(j) * aj1] * b1(o4) = 0.00 * 0.50 = 0.00189
    Psi4(1) = argmax[delta3(j) * aj1] = 0
    delta4(2) = max[delta3(j) * aj2] * b2(o4) = 0.01 * 0.60 = 0.00302
    Psi4(2) = argmax[delta3(j) * aj2] = 1
    delta4(3) = max[delta3(j) * aj3] * b3(o4) = 0.01 * 0.30 = 0.00220
    Psi4(3) = argmax[delta3(j) * aj3] = 2
    
    所有路径的终结点最大的概率值：
    P = 0.003024
    
    最优路径的终结点：
    i4 = argmax[deltaT(i)] = 2
    
    最优路径的其他结点：
    i3 = Psi4(i4) = 2
    i2 = Psi3(i3) = 2
    i1 = Psi2(i2) = 3
    
    最优路径是： 3->2->2->2
    

所以，最优路径$I^*=(i_1^*,i_2^*,i_3^*,i_4^*)=(3,2,2,2)$。

## 习题10.4
&emsp;&emsp;试用前向概率和后向概率推导$$P(O|\lambda)=\sum_{i=1}^N\sum_{j=1}^N\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j),\quad t=1,2,\cdots,T-1$$

**解答：**

**解答思路：**
1. 将$P(O|\lambda)$按照定义展开，即$P(O|\lambda) = P(o_1,o_2,...,o_T|\lambda)$
2. 假设在时刻$t$状态为$q_i$的条件下，将概率拆分为两个条件概率（前向概率、后向概率）
3. 在后向概率中，假设在时刻$t+1$状态为$q_j$，继续拆分为两个条件概率（$t+1$时刻的概率和$t+2$至$T$的概率）
4. 将$t+1$时刻的概率拆分为$t+1$时刻的观测概率和状态转移概率
5. 按照前向概率和后向概率定义，使用$\alpha,\beta,a,b$来表示

**解答步骤：**  

**第1步：$P(O|\lambda)$展开推导**

$$\begin{aligned}
P(O|\lambda)
&= P(o_1,o_2,...,o_T|\lambda) \\
&= \sum_{i=1}^N P(o_1,..,o_t,i_t=q_i|\lambda) P(o_{t+1},..,o_T|i_t=q_i,\lambda) \\
&= \sum_{i=1}^N \sum_{j=1}^N P(o_1,..,o_t,i_t=q_i|\lambda) P(o_{t+1},i_{t+1}=q_j|i_t=q_i,\lambda)P(o_{t+2},..,o_T|i_{t+1}=q_j,\lambda) \\
&= \sum_{i=1}^N \sum_{j=1}^N [P(o_1,..,o_t,i_t=q_i|\lambda) P(o_{t+1}|i_{t+1}=q_j,\lambda) P(i_{t+1}=q_j|i_t=q_i,\lambda) \\
& \quad \quad \quad \quad P(o_{t+2},..,o_T|i_{t+1}=q_j,\lambda)] \\
\end{aligned}$$

**第2步：前向概率和后向概率**

根据书中第198页前向概率：
> &emsp;&emsp;给定隐马尔可夫模型$\lambda$，定义到时刻$t$部分观测序列为$o_1, o_2, \cdots, o_t$且状态为$q_i$的概率为前向概率，记作  
> $$
\alpha_t(i) = P(o_1, o_2, \cdots, o_t, i_t=q_i | \lambda)
$$
> 可以递推地求得前向概率$\alpha_t(i)$及观测序列概率$P(O|\lambda)$

根据书中第201页后向概率：
> &emsp;&emsp;给定隐马尔可夫模型$\lambda$，定义在时刻$t$状态为$q_i$的条件下，从$t+1$到$T$的部分观测序列为$o_{t+1}, o_{t+2}, \cdots, o_T$的后向概率，记作  
> $$
\beta_t(i) = P(o_{t+1}, o_{t+2}, \cdots, o_T| | i_t=q_i , \lambda)
$$
> 可以用递推的方法求得后向概率$\beta_t(i)$及观测序列概率$P(O|\lambda)$

**第3步：用$\alpha,\beta,a,b$表示**

根据书中第193~194页状态转移概率矩阵公式(10.2)和观测概率矩阵公式(10.4)的定义：
> $$
a_{ij} = P(i_{t+1}=q_j | i_t = q_i), \quad i=1,2,\cdots,N; \quad j = 1,2,\cdots, N \\
b_j(k) = P(o_t = v_k | i_t = q_j), \quad k=1,2, \cdots, M; \quad j=1,2,\cdots,N
$$

则
$$\begin{aligned}
P(O|\lambda) 
&= \sum_{i=1}^N \sum_{j=1}^N [P(o_1,..,o_t,i_t=q_i|\lambda) P(o_{t+1}|i_{t+1}=q_j,\lambda) P(i_{t+1}=q_j|i_t=q_i,\lambda) \\
& \quad \quad \quad \quad P(o_{t+2},..,o_T|i_{t+1}=q_j,\lambda)] \\
&= \sum_{i=1}^N \sum_{j=1}^N \alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j), \quad t=1,2,...,T-1 \\
\end{aligned}
$$
命题得证。

## 习题10.5

&emsp;&emsp;比较维特比算法中变量$\delta$的计算和前向算法中变量$\alpha$的计算的主要区别。

**解答：** 

**解答思路：**
1. 列出维特比算法中变量$\delta$的计算；
2. 列出前向算法中变量$\alpha$的计算；
3. 比较两个变量计算的主要区别。

**解答步骤：**

**第1步：维特比算法中变量$\delta$的计算**

&emsp;&emsp;根据书中第209页维特比算法：
> （1）初始化
> $$
\delta_1(i)=\pi_ib_i(o_1),i=1,2,\cdots,N
$$
> （2）递推，对$t=2,3,\cdots,T$
> $$
\delta_t(i)=\max_{1 \leqslant j \leqslant N} [\delta_{t-1}(j)a_{ji}]b_i(o_t), i=1,2,\cdots,N
$$  

**第2步：前向算法中变量$\alpha$的计算**

&emsp;&emsp;根据书中第198页观测序列概率的前向算法：
> （1）初值
> $$
\alpha_1(i)=\pi_ib_i(o_i),i=1,2,\cdots,N
$$
>（2）递推，对$t=1,2,\cdots,T-1$：
> $$
\alpha_{t+1}(i)=\left[\sum_{j=1}^N \alpha_t(j) a_{ji} \right]b_i(o_{t+1})，i=1,2,\cdots,N
$$

**第3步：比较两个变量计算的主要区别**

通过比较两个变量的计算，主要区别包括计算目标不同和关注对象不同两个方面：
1. 计算目标不同
  - 前向算法：计算长度为$t$的观测序列概率，以某一时刻某一状态为最终状态的概率；
  - 维特比算法：计算长度为$t$的观测序列概率，选择当前观测序列的最大可能概率；
2. 关注对象不同
  - 前向算法：包含到$t-1$时刻所有可能的观测序列概率和，其中不变的是观测序列上的最终状态；
  - 维特比算法：包含出现观测序列中所有$t$个时刻相应观测的最大概率，其中不变的是整条观测序列。
