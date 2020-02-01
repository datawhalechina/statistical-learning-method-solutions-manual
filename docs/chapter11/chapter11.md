# 第11章条件随机场-习题

## 习题11.1
&emsp;&emsp;写出图11.3中无向图描述的概率图模型的因子分解式。
<br/><center>
<img style="border-radius: 0.3125em;box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="chapter11/11-1-Maximal-Clique.png"><br><div style="color:orange; border-bottom: 1px solid #d9d9d9;display: inline-block;color: #000;padding: 2px;">图11.3 无向图的团和最大团</div></center>

**解答：**  
&emsp;&emsp;图11.3表示由4个结点组成的无向图。图中由2个结点组成的团有5个：$\{Y_1,Y_2\},\{Y_2,Y_3\},\{Y_3,Y_4\},\{Y_4,Y_2\}$和$\{Y_1,Y_3\}$，有2个最大团：$\{Y_1,Y_2,Y_3\}$和$\{Y_2,Y_3,Y_4\}$，而$\{Y_1,Y_2,Y_3,Y_4\}$不是一个团，因为$Y_1$和$Y_4$没有边连接。  
&emsp;&emsp;根据概率图模型的因子分解定义：将概率无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积形式的操作。公式在书中(11.5)，(11.6)。  
$$P(Y)=\frac{\Psi_{(1,2,3)}(Y_{(1,2,3)})\cdot\Psi_{(2,3,4)}(Y_{(2,3,4)})}{\displaystyle \sum_Y \left[ \Psi_{(1,2,3)}(Y_{(1,2,3)})\cdot\Psi_{(2,3,4)}(Y_{(2,3,4)})\right]}$$

## 习题11.2

&emsp;&emsp;证明$Z(x)=a_n^T(x) \cdot \boldsymbol{1} = \boldsymbol{1}^T\cdot\beta_1(x)$，其中$\boldsymbol{1}$是元素均为1的$m$维列向量。

**解答：**  
**第1步：**证明$Z(x)=a_n^T(x) \cdot \boldsymbol{1}$  
根据条件随机场的矩阵形式：$$(M_{n+1}(x))_{i,j}=\begin{cases}
1,&j=\text{stop}\\
0,&\text{otherwise}
\end{cases}$$根据前向向量的定义：$$\alpha_0(y|x)=\begin{cases}
1,&y=\text{start} \\
0,&\text{otherwise}
\end{cases}$$  
$\begin{aligned}
\therefore Z_n(x) 
&= \left(M_1(x)M_2(x){\cdots}M_{n+1}(x)\right)_{(\text{start},\text{stop})} \\
&= \alpha_0(x)^T M_1(x)M_2(x){\cdots}M_n(x) \cdot 1\\
&=\alpha_n(x)^T\cdot \boldsymbol{1}
\end{aligned}$  

-----

**第2步：**证明$Z(x)=\boldsymbol{1}^T \cdot \beta_1(x)$  
根据条件随机场的矩阵形式：$$(M_{n+1}(x))_{i,j}=\begin{cases}
1,&j=\text{stop}\\
0,&\text{otherwise}
\end{cases}$$根据后向向量定义：$$\beta_{n+1}(y_{n+1}|x)=
\begin{cases}
1,& y_{n+1}=\text{stop} \\
0,& \text{otherwise}
\end{cases}$$  
$\begin{aligned}
\therefore Z_n(x)
&= (M_1(x)M_2(x) \cdots M_{n+1}(x))_{(\text{start},\text{stop})} \\
&= (M_1(x)M_2(x) \cdots M_n(x) \beta_{n+1}(x))_{\text{start}} \\
&=(\beta_1(x))_{\text{start}} \\
&=\boldsymbol{1}^T \cdot \beta_1(x)
\end{aligned}$  
综上所述：$Z(x)=a_n^T(x) \cdot \boldsymbol{1} = \boldsymbol{1}^T \cdot \beta_1(x)$，命题得证。  

## 习题11.3
&emsp;&emsp;写出条件随机场模型学习的梯度下降法。

**解答：**  
条件随机场的对数极大似然函数为：$$L(w)=\sum^N_{j=1} \sum^K_{k=1} w_k f_k(y_j,x_j)-\sum^N_{j=1} \log{Z_w(x_j)}$$梯度下降算法的目标函数是$f(w)=-L(w)$  
目标函数的梯度为：$$g(w)=\frac{\nabla{f(w)}}{\nabla{w}}=\left(\frac{\partial{f(w)}}{\partial{w_1}},\frac{\partial{f(w)}}{\partial{w_2}},\cdots,\frac{\partial{f(w)}}{\partial{w_k}}\right)$$其中$$\begin{aligned}
\frac{\partial{f(w)}}{\partial{w_i}}
&= -\sum^N_{j=1} w_i f_i(y_j,x_j) + \sum^N_{j=1} \frac{1}{Z_w(x_j)} \cdot \frac{\partial{Z_w(x_j)}}{\partial{w_i}}\\
&= -\sum^N_{j=1}w_if_i(y_j,x_j)+\sum^N_{j=1}\frac{1}{Z_w(x_j)}\sum_y(\exp{\sum^K_{k=1}w_kf_k(y,x_j))}w_if_i(y,x_j)
\end{aligned}$$  
根据梯度下降算法：  
1. 取初始值$w^{(0)} \in \mathbf{R}^n$，置$k=0$  
2. 计算$f(w^{(k)})$  
3. 计算梯度$g_k=g(w^{(k)})$，当$\|g_k\|<\varepsilon$时，停止迭代，令$w^*=w^{(k)}$；否则令$p_k=-g(w^{(k)})$，求$\lambda_k$，使$$
f(w^{(k)}+\lambda_k p_k)=\min_{\lambda \geqslant 0}{f(w^{(k)}+\lambda p_k)}$$  
4. 置$w^{(k+1)}=w^{(k)}+\lambda_k p_k$，计算$f(w^{(k+1)})$  
当$\|f(w^{(k+1)})-f(w^{(k)})\| < \epsilon$或$\|w^{(k+1)}-w^{(k)}\| < \epsilon$时，停止迭代，令$w^*=w^{(k+1)}$  
5. 否则，置$k=k+1$，转(3).

## 习题11.4

参考图11.6的状态路径图，假设随机矩阵$M_1(x),M_2(x),M_3(x),M_4(x)$分别是
$$M_1(x)=\begin{bmatrix}0&0\\0.5&0.5\end{bmatrix} ,
M_2(x)=\begin{bmatrix}0.3&0.7\\0.7&0.3\end{bmatrix}$$
$$
M_3(x)=\begin{bmatrix}0.5&0.5\\0.6&0.4\end{bmatrix},
M_4(x)=\begin{bmatrix}0&1\\0&1\end{bmatrix}$$
求以$start=2$为起点$stop=2$为终点的所有路径的状态序列$y$的概率及概率最大的状态序列。

**解答：**


```python
import numpy as np

# 创建随机矩阵
M1 = [[0,0],[0.5,0.5]] 
M2 = [[0.3, 0.7], [0.7,0.3]]
M3 = [[0.5, 0.5], [0.6, 0.4]]
M4 = [[0, 1], [0, 1]]
M = [M1, M2, M3, M4]
print(M)
```

    [[[0, 0], [0.5, 0.5]], [[0.3, 0.7], [0.7, 0.3]], [[0.5, 0.5], [0.6, 0.4]], [[0, 1], [0, 1]]]
    


```python
# 生成路径
path = [2]
for i in range(1,4):
    paths = []
    for _, r in enumerate(path):
        temp = np.transpose(r)
        paths.append(np.append(temp, 1))
        paths.append(np.append(temp, 2))
    path = paths.copy()

path = [np.append(r, 2) for _, r in enumerate(path)]
print(path)
```

    [array([2, 1, 1, 1, 2]), array([2, 1, 1, 2, 2]), array([2, 1, 2, 1, 2]), array([2, 1, 2, 2, 2]), array([2, 2, 1, 1, 2]), array([2, 2, 1, 2, 2]), array([2, 2, 2, 1, 2]), array([2, 2, 2, 2, 2])]
    


```python
# 计算概率

pr = []
for _, row in enumerate(path):
    p = 1
    for i in range(len(row)-1):
        a = row[i]
        b = row[i+1]
        p *= M[i][a-1][b-1]
    pr.append((row.tolist(), p))
pr = sorted(pr, key=lambda x : x[1], reverse=True)
print(pr)
```

    [([2, 1, 2, 1, 2], 0.21), ([2, 2, 1, 1, 2], 0.175), ([2, 2, 1, 2, 2], 0.175), ([2, 1, 2, 2, 2], 0.13999999999999999), ([2, 2, 2, 1, 2], 0.09), ([2, 1, 1, 1, 2], 0.075), ([2, 1, 1, 2, 2], 0.075), ([2, 2, 2, 2, 2], 0.06)]
    


```python
# 打印结果
print("以start=2为起点stop=2为终点的所有路径的状态序列y的概率为：")
for path, p in pr:
    print("    路径为：" + "->".join([ str(x) for x in path]) ,end=" ")
    print("概率为：" + str(p))
print("概率[" + str(pr[0][1]) +"]最大的状态序列为：" + "->".join([str(x) for x in pr[0][0]]), end = " ")
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
    概率[0.21]最大的状态序列为：2->1->2->1->2 
