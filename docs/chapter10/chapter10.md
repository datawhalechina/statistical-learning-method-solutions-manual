
## 第10章隐马尔可夫模型-习题

### 习题10.1
&emsp;&emsp;给定盒子和球组成的隐马尔可夫模型$\lambda=(A,B,\pi)$，其中，$$A=\left[\begin{array}{ccc}0.5&0.2&0.3\\0.3&0.5&0.2\\0.2&0.3&0.5\end{array}\right], \quad B=\left[\begin{array}{cc}0.5&0.5\\0.4&0.6\\0.7&0.3\end{array}\right], \quad \pi=(0.2,0.4,0.4)^T$$设$T=4,O=(红,白,红,白)$，试用后向算法计算$P(O|\lambda)$。

**解答：**


```python
import numpy as np

class HiddenMarkov:
    def __init__(self):
        self.alphas = None
        self.forward_P = None
        self.betas = None
        self.backward_P = None
    
    # 前向算法
    def forward(self, Q, V, A, B, O, PI):
        # 状态序列的大小
        N = len(Q)  
        # 观测序列的大小
        M = len(O)  
        # 初始化前向概率alpha值
        alphas = np.zeros((N, M))
        # 时刻数=观测序列数
        T = M  
        # 遍历每一个时刻，计算前向概率alpha值
        for t in range(T):  
            # 得到序列对应的索引
            indexOfO = V.index(O[t])
            # 遍历状态序列
            for i in range(N):
                # 初始化alpha初值
                if t == 0: 
                    # P176 公式(10.15)
                    alphas[i][t] = PI[t][i] * B[i][indexOfO]  
                    print('alpha1(%d) = p%db%db(o1) = %f' % (i+1, i, i, alphas[i][t]))
                else:
                    # P176 公式(10.16)
                    alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas], [a[i] for a in A]) * B[i][indexOfO]  
                    print('alpha%d(%d) = [sigma alpha%d(i)ai%d]b%d(o%d) = %f' % (t+1, i+1, t - 1, i, i, t, alphas[i][t]))
        # P176 公式(10.17)
        self.forward_P = np.sum([alpha[M - 1] for alpha in alphas]) 
        self.alphas = alphas        
        
    # 后向算法
    def backward(self, Q, V, A, B, O, PI):  
        # 状态序列的大小
        N = len(Q) 
        # 观测序列的大小
        M = len(O) 
        # 初始化后向概率beta值，P178 公式(10.19)
        betas = np.ones((N, M))  
        # 
        for i in range(N):
            print('beta%d(%d) = 1' % (M, i+1))
        # 对观测序列逆向遍历
        for t in range(M - 2, -1, -1):
            # 得到序列对应的索引
            indexOfO = V.index(O[t + 1])  
             # 遍历状态序列
            for i in range(N):
                # P178 公式(10.20)
                betas[i][t] = np.dot(np.multiply(A[i], [b[indexOfO] for b in B]), [beta[t + 1] for beta in betas])
                realT = t + 1
                realI = i + 1
                print('beta%d(%d) = sigma[a%djbj(o%d)beta%d(j)] = (' % (realT, realI, realI, realT + 1, realT + 1),
                      end='')
                for j in range(N):
                    print("%.2f * %.2f * %.2f + " % (A[i][j], B[j][indexOfO], betas[j][t + 1]), end='')
                print("0) = %.3f" % betas[i][t])
        # 取出第一个值
        indexOfO = V.index(O[0])
        self.betas = betas
        # P178 公式(10.21)
        P = np.dot(np.multiply(PI, [b[indexOfO] for b in B]), [beta[0] for beta in betas])
        self.backward_P = P
        print("P(O|lambda) = ", end="")
        for i in range(N):
            print("%.1f * %.1f * %.5f + " % (PI[0][i], B[i][indexOfO], betas[i][0]), end="")
        print("0 = %f" % P)
    
    # 维特比算法
    def viterbi(self, Q, V, A, B, O, PI):
        # 状态序列的大小
        N = len(Q)  
        # 观测序列的大小
        M = len(O)  
        # 初始化daltas
        deltas = np.zeros((N, M))
        # 初始化psis
        psis = np.zeros((N, M))
        # 初始化最优路径矩阵，该矩阵维度与观测序列维度相同
        I = np.zeros((1, M))
        # 遍历观测序列
        for t in range(M):
            # 递推从t=2开始
            realT = t+1
            # 得到序列对应的索引
            indexOfO = V.index(O[t]) 
            for i in range(N):
                realI = i+1
                if t == 0:
                    # P185 算法10.5 步骤(1)
                    deltas[i][t] = PI[0][i] * B[i][indexOfO]
                    psis[i][t] = 0
                    print('delta1(%d) = pi%d * b%d(o1) = %.2f * %.2f = %.2f'%(realI, realI, realI, PI[0][i], B[i][indexOfO], deltas[i][t]))
                    print('psis1(%d) = 0' % (realI))
                else:
                    # # P185 算法10.5 步骤(2)
                    deltas[i][t] = np.max(np.multiply([delta[t-1] for delta in deltas], [a[i] for a in A])) * B[i][indexOfO]
                    print('delta%d(%d) = max[delta%d(j)aj%d]b%d(o%d) = %.2f * %.2f = %.5f'%(realT, realI, realT-1, realI, realI, realT, np.max(np.multiply([delta[t-1] for delta in deltas], [a[i] for a in A])), B[i][indexOfO], deltas[i][t]))
                    psis[i][t] = np.argmax(np.multiply([delta[t-1] for delta in deltas], [a[i] for a in A]))
                    print('psis%d(%d) = argmax[delta%d(j)aj%d] = %d' % (realT, realI, realT-1, realI, psis[i][t]))
        #print(deltas)
        #print(psis)
        # 得到最优路径的终结点
        I[0][M-1] = np.argmax([delta[M-1] for delta in deltas])
        print('i%d = argmax[deltaT(i)] = %d' % (M, I[0][M-1]+1))
        # 递归由后向前得到其他结点
        for t in range(M-2, -1, -1):
            I[0][t] = psis[int(I[0][t+1])][t+1]
            print('i%d = psis%d(i%d) = %d' % (t+1, t+2, t+2, I[0][t]+1))
        # 输出最优路径
        print('最优路径是：',"->".join([str(int(i+1)) for i in I[0]]))
```


```python
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '白'] 
PI = [[0.2, 0.4, 0.4]]

HMM = HiddenMarkov()
HMM.backward(Q, V, A, B, O, PI)
```

    beta4(1) = 1
    beta4(2) = 1
    beta4(3) = 1
    beta3(1) = sigma[a1jbj(o4)beta4(j)] = (0.50 * 0.50 * 1.00 + 0.20 * 0.60 * 1.00 + 0.30 * 0.30 * 1.00 + 0) = 0.460
    beta3(2) = sigma[a2jbj(o4)beta4(j)] = (0.30 * 0.50 * 1.00 + 0.50 * 0.60 * 1.00 + 0.20 * 0.30 * 1.00 + 0) = 0.510
    beta3(3) = sigma[a3jbj(o4)beta4(j)] = (0.20 * 0.50 * 1.00 + 0.30 * 0.60 * 1.00 + 0.50 * 0.30 * 1.00 + 0) = 0.430
    beta2(1) = sigma[a1jbj(o3)beta3(j)] = (0.50 * 0.50 * 0.46 + 0.20 * 0.40 * 0.51 + 0.30 * 0.70 * 0.43 + 0) = 0.246
    beta2(2) = sigma[a2jbj(o3)beta3(j)] = (0.30 * 0.50 * 0.46 + 0.50 * 0.40 * 0.51 + 0.20 * 0.70 * 0.43 + 0) = 0.231
    beta2(3) = sigma[a3jbj(o3)beta3(j)] = (0.20 * 0.50 * 0.46 + 0.30 * 0.40 * 0.51 + 0.50 * 0.70 * 0.43 + 0) = 0.258
    beta1(1) = sigma[a1jbj(o2)beta2(j)] = (0.50 * 0.50 * 0.25 + 0.20 * 0.60 * 0.23 + 0.30 * 0.30 * 0.26 + 0) = 0.112
    beta1(2) = sigma[a2jbj(o2)beta2(j)] = (0.30 * 0.50 * 0.25 + 0.50 * 0.60 * 0.23 + 0.20 * 0.30 * 0.26 + 0) = 0.122
    beta1(3) = sigma[a3jbj(o2)beta2(j)] = (0.20 * 0.50 * 0.25 + 0.30 * 0.60 * 0.23 + 0.50 * 0.30 * 0.26 + 0) = 0.105
    P(O|lambda) = 0.2 * 0.5 * 0.11246 + 0.4 * 0.4 * 0.12174 + 0.4 * 0.7 * 0.10488 + 0 = 0.060091
    

可得$P(O|\lambda) = 0.060091$

### 习题10.2

&emsp;&emsp;给定盒子和球组成的隐马尔可夫模型$\lambda=(A,B,\pi)$，其中，$$A=\left[\begin{array}{ccc}0.5&0.1&0.4\\0.3&0.5&0.2\\0.2&0.2&0.6\end{array}\right], \quad B=\left[\begin{array}{cc}0.5&0.5\\0.4&0.6\\0.7&0.3\end{array}\right], \quad \pi=(0.2,0.3,0.5)^T$$设$T=8,O=(红,白,红,红,白,红,白,白)$，试用前向后向概率计算$P(i_4=q_3|O,\lambda)$

**解答：**


```python
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '红', '白', '红', '白', '白']
PI = [[0.2, 0.3, 0.5]]

HMM = HiddenMarkov()
HMM.forward(Q, V, A, B, O, PI)
HMM.backward(Q, V, A, B, O, PI)
```

    alpha1(1) = p0b0b(o1) = 0.100000
    alpha1(2) = p1b1b(o1) = 0.120000
    alpha1(3) = p2b2b(o1) = 0.350000
    alpha2(1) = [sigma alpha0(i)ai0]b0(o1) = 0.078000
    alpha2(2) = [sigma alpha0(i)ai1]b1(o1) = 0.084000
    alpha2(3) = [sigma alpha0(i)ai2]b2(o1) = 0.082200
    alpha3(1) = [sigma alpha1(i)ai0]b0(o2) = 0.040320
    alpha3(2) = [sigma alpha1(i)ai1]b1(o2) = 0.026496
    alpha3(3) = [sigma alpha1(i)ai2]b2(o2) = 0.068124
    alpha4(1) = [sigma alpha2(i)ai0]b0(o3) = 0.020867
    alpha4(2) = [sigma alpha2(i)ai1]b1(o3) = 0.012362
    alpha4(3) = [sigma alpha2(i)ai2]b2(o3) = 0.043611
    alpha5(1) = [sigma alpha3(i)ai0]b0(o4) = 0.011432
    alpha5(2) = [sigma alpha3(i)ai1]b1(o4) = 0.010194
    alpha5(3) = [sigma alpha3(i)ai2]b2(o4) = 0.011096
    alpha6(1) = [sigma alpha4(i)ai0]b0(o5) = 0.005497
    alpha6(2) = [sigma alpha4(i)ai1]b1(o5) = 0.003384
    alpha6(3) = [sigma alpha4(i)ai2]b2(o5) = 0.009288
    alpha7(1) = [sigma alpha5(i)ai0]b0(o6) = 0.002811
    alpha7(2) = [sigma alpha5(i)ai1]b1(o6) = 0.002460
    alpha7(3) = [sigma alpha5(i)ai2]b2(o6) = 0.002535
    alpha8(1) = [sigma alpha6(i)ai0]b0(o7) = 0.001325
    alpha8(2) = [sigma alpha6(i)ai1]b1(o7) = 0.001211
    alpha8(3) = [sigma alpha6(i)ai2]b2(o7) = 0.000941
    beta8(1) = 1
    beta8(2) = 1
    beta8(3) = 1
    beta7(1) = sigma[a1jbj(o8)beta8(j)] = (0.50 * 0.50 * 1.00 + 0.10 * 0.60 * 1.00 + 0.40 * 0.30 * 1.00 + 0) = 0.430
    beta7(2) = sigma[a2jbj(o8)beta8(j)] = (0.30 * 0.50 * 1.00 + 0.50 * 0.60 * 1.00 + 0.20 * 0.30 * 1.00 + 0) = 0.510
    beta7(3) = sigma[a3jbj(o8)beta8(j)] = (0.20 * 0.50 * 1.00 + 0.20 * 0.60 * 1.00 + 0.60 * 0.30 * 1.00 + 0) = 0.400
    beta6(1) = sigma[a1jbj(o7)beta7(j)] = (0.50 * 0.50 * 0.43 + 0.10 * 0.60 * 0.51 + 0.40 * 0.30 * 0.40 + 0) = 0.186
    beta6(2) = sigma[a2jbj(o7)beta7(j)] = (0.30 * 0.50 * 0.43 + 0.50 * 0.60 * 0.51 + 0.20 * 0.30 * 0.40 + 0) = 0.241
    beta6(3) = sigma[a3jbj(o7)beta7(j)] = (0.20 * 0.50 * 0.43 + 0.20 * 0.60 * 0.51 + 0.60 * 0.30 * 0.40 + 0) = 0.176
    beta5(1) = sigma[a1jbj(o6)beta6(j)] = (0.50 * 0.50 * 0.19 + 0.10 * 0.40 * 0.24 + 0.40 * 0.70 * 0.18 + 0) = 0.106
    beta5(2) = sigma[a2jbj(o6)beta6(j)] = (0.30 * 0.50 * 0.19 + 0.50 * 0.40 * 0.24 + 0.20 * 0.70 * 0.18 + 0) = 0.101
    beta5(3) = sigma[a3jbj(o6)beta6(j)] = (0.20 * 0.50 * 0.19 + 0.20 * 0.40 * 0.24 + 0.60 * 0.70 * 0.18 + 0) = 0.112
    beta4(1) = sigma[a1jbj(o5)beta5(j)] = (0.50 * 0.50 * 0.11 + 0.10 * 0.60 * 0.10 + 0.40 * 0.30 * 0.11 + 0) = 0.046
    beta4(2) = sigma[a2jbj(o5)beta5(j)] = (0.30 * 0.50 * 0.11 + 0.50 * 0.60 * 0.10 + 0.20 * 0.30 * 0.11 + 0) = 0.053
    beta4(3) = sigma[a3jbj(o5)beta5(j)] = (0.20 * 0.50 * 0.11 + 0.20 * 0.60 * 0.10 + 0.60 * 0.30 * 0.11 + 0) = 0.043
    beta3(1) = sigma[a1jbj(o4)beta4(j)] = (0.50 * 0.50 * 0.05 + 0.10 * 0.40 * 0.05 + 0.40 * 0.70 * 0.04 + 0) = 0.026
    beta3(2) = sigma[a2jbj(o4)beta4(j)] = (0.30 * 0.50 * 0.05 + 0.50 * 0.40 * 0.05 + 0.20 * 0.70 * 0.04 + 0) = 0.023
    beta3(3) = sigma[a3jbj(o4)beta4(j)] = (0.20 * 0.50 * 0.05 + 0.20 * 0.40 * 0.05 + 0.60 * 0.70 * 0.04 + 0) = 0.027
    beta2(1) = sigma[a1jbj(o3)beta3(j)] = (0.50 * 0.50 * 0.03 + 0.10 * 0.40 * 0.02 + 0.40 * 0.70 * 0.03 + 0) = 0.015
    beta2(2) = sigma[a2jbj(o3)beta3(j)] = (0.30 * 0.50 * 0.03 + 0.50 * 0.40 * 0.02 + 0.20 * 0.70 * 0.03 + 0) = 0.012
    beta2(3) = sigma[a3jbj(o3)beta3(j)] = (0.20 * 0.50 * 0.03 + 0.20 * 0.40 * 0.02 + 0.60 * 0.70 * 0.03 + 0) = 0.016
    beta1(1) = sigma[a1jbj(o2)beta2(j)] = (0.50 * 0.50 * 0.01 + 0.10 * 0.60 * 0.01 + 0.40 * 0.30 * 0.02 + 0) = 0.006
    beta1(2) = sigma[a2jbj(o2)beta2(j)] = (0.30 * 0.50 * 0.01 + 0.50 * 0.60 * 0.01 + 0.20 * 0.30 * 0.02 + 0) = 0.007
    beta1(3) = sigma[a3jbj(o2)beta2(j)] = (0.20 * 0.50 * 0.01 + 0.20 * 0.60 * 0.01 + 0.60 * 0.30 * 0.02 + 0) = 0.006
    P(O|lambda) = 0.2 * 0.5 * 0.00633 + 0.3 * 0.4 * 0.00685 + 0.5 * 0.7 * 0.00578 + 0 = 0.003477
    

可知，$\displaystyle P(i_4=q_3|O,\lambda)=\frac{P(i_4=q_3,O|\lambda)}{P(O|\lambda)}=\frac{\alpha_4(3)\beta_4(3)}{P(O|\lambda)}$


```python
print("alpha4(3)=", HMM.alphas[3-1][4-1])
print("beta4(3)=", HMM.betas[3-1][4-1])
print("P(O|lambda)=", HMM.backward_P[0])
result = (HMM.alphas[3-1][4-1] * HMM.betas[3-1][4-1]) / HMM.backward_P[0]
print("P(i4=q3|O,lambda) =",result)
```

    alpha4(3)= 0.043611119999999996
    beta4(3)= 0.04280618
    P(O|lambda)= 0.0034767094492824
    P(i4=q3|O,lambda) = 0.5369518160647322
    

### 习题10.3

&emsp;&emsp;在习题10.1中，试用维特比算法求最优路径$I^*=(i_1^*,i_2^*,i_3^*,i_4^*)$。


```python
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '白'] 
PI = [[0.2, 0.4, 0.4]]

HMM = HiddenMarkov()
HMM.viterbi(Q, V, A, B, O, PI)
```

    delta1(1) = pi1 * b1(o1) = 0.20 * 0.50 = 0.10
    psis1(1) = 0
    delta1(2) = pi2 * b2(o1) = 0.40 * 0.40 = 0.16
    psis1(2) = 0
    delta1(3) = pi3 * b3(o1) = 0.40 * 0.70 = 0.28
    psis1(3) = 0
    delta2(1) = max[delta1(j)aj1]b1(o2) = 0.06 * 0.50 = 0.02800
    psis2(1) = argmax[delta1(j)aj1] = 2
    delta2(2) = max[delta1(j)aj2]b2(o2) = 0.08 * 0.60 = 0.05040
    psis2(2) = argmax[delta1(j)aj2] = 2
    delta2(3) = max[delta1(j)aj3]b3(o2) = 0.14 * 0.30 = 0.04200
    psis2(3) = argmax[delta1(j)aj3] = 2
    delta3(1) = max[delta2(j)aj1]b1(o3) = 0.02 * 0.50 = 0.00756
    psis3(1) = argmax[delta2(j)aj1] = 1
    delta3(2) = max[delta2(j)aj2]b2(o3) = 0.03 * 0.40 = 0.01008
    psis3(2) = argmax[delta2(j)aj2] = 1
    delta3(3) = max[delta2(j)aj3]b3(o3) = 0.02 * 0.70 = 0.01470
    psis3(3) = argmax[delta2(j)aj3] = 2
    delta4(1) = max[delta3(j)aj1]b1(o4) = 0.00 * 0.50 = 0.00189
    psis4(1) = argmax[delta3(j)aj1] = 0
    delta4(2) = max[delta3(j)aj2]b2(o4) = 0.01 * 0.60 = 0.00302
    psis4(2) = argmax[delta3(j)aj2] = 1
    delta4(3) = max[delta3(j)aj3]b3(o4) = 0.01 * 0.30 = 0.00220
    psis4(3) = argmax[delta3(j)aj3] = 2
    i4 = argmax[deltaT(i)] = 2
    i3 = psis4(i4) = 2
    i2 = psis3(i3) = 2
    i1 = psis2(i2) = 3
    最优路径是： 3->2->2->2
    

### 习题10.4
&emsp;&emsp;试用前向概率和后向概率推导$$P(O|\lambda)=\sum_{i=1}^N\sum_{j=1}^N\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j),\quad t=1,2,\cdots,T-1$$

**解答：**  
$$\begin{aligned}
P(O|\lambda)
&= P(o_1,o_2,...,o_T|\lambda) \\
&= \sum_{i=1}^N P(o_1,..,o_t,i_t=q_i|\lambda) P(o_{t+1},..,o_T|i_t=q_i,\lambda) \\
&= \sum_{i=1}^N \sum_{j=1}^N P(o_1,..,o_t,i_t=q_i|\lambda) P(o_{t+1},i_{t+1}=q_j|i_t=q_i,\lambda)P(o_{t+2},..,o_T|i_{t+1}=q_j,\lambda) \\
&= \sum_{i=1}^N \sum_{j=1}^N P(o_1,..,o_t,i_t=q_i|\lambda) P(o_{t+1}|i_{t+1}=q_j,\lambda) P(i_{t+1}=q_j|i_t=q_i,\lambda) P(o_{t+2},..,o_T|i_{t+1}=q_j,\lambda) \\
&= \sum_{i=1}^N \sum_{j=1}^N \alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j),{\quad}t=1,2,...,T-1
\end{aligned}$$
命题得证。

### 习题10.5

&emsp;&emsp;比较维特比算法中变量$\delta$的计算和前向算法中变量$\alpha$的计算的主要区别。

**解答：**  
**前向算法：**  
1. 初值$$\alpha_1(i)=\pi_ib_i(o_i),i=1,2,\cdots,N$$
2. 递推，对$t=1,2,\cdots,T-1$：$$\alpha_{t+1}(i)=\left[\sum_{j=1}^N \alpha_t(j) a_{ji} \right]b_i(o_{t+1})，i=1,2,\cdots,N$$

**维特比算法：**  
1. 初始化$$\delta_1(i)=\pi_ib_i(o_1),i=1,2,\cdots,N$$
2. 递推，对$t=2,3,\cdots,T$$$\delta_t(i)=\max_{1 \leqslant j \leqslant N} [\delta_{t-1}(j)a_{ji}]b_i(o_t), i=1,2,\cdots,N$$  

&emsp;&emsp;由上面算法可知，计算变量$\alpha$的时候直接对上个的结果进行数值计算，而计算变量$\delta$需要在上个结果计算的基础上选择最大值
