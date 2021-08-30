# 第4章 朴素贝叶斯法

## 习题4.1
&emsp;&emsp;用极大似然估计法推出朴素贝叶斯法中的概率估计公式(4.8)及公式 (4.9)。

**解答：**  

**解答思路：**
1. 极大似然估计的一般步骤（详见习题1.1第3步）
2. 证明公式4.8：根据输出空间$\mathcal{Y}$的随机变量$Y$满足独立同分布，列出似然函数，求解概率$P(Y=c_k)$的值；
3. 证明公式4.9：证明同公式4.8。

**解答步骤：**

**第1步：极大似然估计的一般步骤**  
> 参考Wiki：https://en.wikipedia.org/wiki/Maximum_likelihood_estimation   
> 1. 写出随机变量的概率分布函数；  
> 2. 写出似然函数；
> 3. 对似然函数取对数，得到对数似然函数，并进行化简；
> 4. 对参数进行求导，并令导数等于0；
> 5. 求解似然函数方程，得到参数的值。

**第2步：证明公式(4.8)** 
$$
\displaystyle P(Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k)}{N}, \quad k=1,2,\ldots,K
$$   

&emsp;&emsp;根据书中第59页，朴素贝叶斯法的基本方法：
> &emsp;&emsp;设输入空间$\mathcal{X} \subseteq R^n$为$n$维向量的集合，输出空间为类标记集合$\mathcal{Y}=\{c_1,c_2,\ldots,c_k\}$。输入为特征向量$x \in \mathcal{X}$，输出为类标记$y \in \mathcal{Y}$。$X$是定义在输入空间$\mathcal{X}$上的随机向量，$Y$是定义在输出空间$\mathcal{Y}$上的随机变量。$P(X,Y)$是$X$和$Y$的联合概率分布。训练数据集
> $$
T=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}
$$
> 由$P(X,Y)$独立同分布产生。  

&emsp;&emsp;根据上述定义，$Y={y_1,y_2,\ldots,y_N}$满足独立同分布，假设$P(Y=c_k)$概率为$p$，其中$c_k$在随机变量$Y$中出现的次数$\displaystyle m=\sum_{i=1}^NI(y_i=c_k)$，可得似然函数为：
$$
\begin{aligned} L(p|Y) &= f(Y|p) \\
&= C_N^m p^m (1-p)^{N-m}
\end{aligned}
$$
对似然函数取对数，得到对数似然函数为：
$$
\begin{aligned} \displaystyle \log L(p|Y) &= \log C_N^m p^m (1-p)^{N-m} \\
&= C_N^m \left[\log(p^m) + \log\left( (1-p)^{N-m} \right)\right] \\
&= C_N^m \left[ m\log p + (N-m)\log (1-p)\right]
\end{aligned}
$$
求解参数$p$：
$$
\begin{aligned}
\hat{p} &= \mathop{\arg\max} \limits_{p} L(p|Y) \\
&= \mathop{\arg\max} \limits_{p} C_N^m \left[ m\log p + (N-m)\log (1-p)\right]
\end{aligned}
$$
对参数$p$求导，并求解导数为0时的$p$值：
$$
\begin{aligned}
\frac{\partial \log L(p)}{\partial p} &= C_N^m \left[ \frac{m}{p} - \frac{N-m}{1-p} \right] \\
&= C_N^m \left[ \frac{m(1-p) - p(N-m)}{p(1-p)} \right] \\
&= C_N^m \frac{m-Np}{p(1-p)} = 0
\end{aligned}
$$
从上式可得，$m-Np=0$，即$\displaystyle P(Y=c_k)=p=\frac{m}{N}$  
综上所述，$\displaystyle P(Y=c_k)=p=\frac{m}{N}=\frac{\displaystyle \sum_{i=1}^N I(y_i=c_k)}{N}$，公式(4.8)得证。

**第3步：证明公式(4.9)** 
$$\displaystyle P(X^{(j)}=a_{jl}|Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)}{\displaystyle \sum_{i=1}^N I(y_i=c_k)} \\
j=1,2,\ldots,n; \quad l = 1,2,\ldots,S_j; \quad k = 1,2,\dots,K
$$

&emsp;&emsp;根据书中第60页，朴素贝叶斯法的条件独立性假设：
> &emsp;&emsp;朴素贝叶斯法对条件概率分布作了条件独立性的假设。由于这是一个较强的假设，朴素贝叶斯法也由此得名。具体地，条件独立性假设是：
> $$
\begin{aligned}
P(X=x|Y=c_k) &= P(X^{(1)}=x^{(1)},\ldots,X^{(n)}=x^{(n)}|Y=c_k) \\
&= \prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)
\end{aligned}
$$

&emsp;&emsp;根据上述定义，在条件$Y=c_k$下，随机变量$X$满足条件独立性，假设$P(X^{(j)}=a_{jl}|Y=c_k)$概率为$p$，其中$c_k$在随机变量$Y$中出现的次数$\displaystyle m=\sum_{i=1}^NI(y_i=c_k)$，$y=c_k$和$x^{(j)}=a_{jl}$同时出现的次数$\displaystyle q=\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)$，可得似然函数为：  
$$
\begin{aligned} L(p|X,Y) &= f(X,Y|p) \\
&= C_m^q p^q (1-p)^{m-q}
\end{aligned}
$$
与第2步推导过程类似，可求解得到$\displaystyle p=\frac{q}{m}$  
综上所述，$\displaystyle P(X^{(j)}=a_{jl}|Y=c_k)=p=\frac{q}{m}=\frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)}{\displaystyle \sum_{i=1}^N I(y_i=c_k)}$，公式(4.9)得证。

## 习题4.2
&emsp;&emsp;用贝叶斯估计法推出朴素贝叶斯法中的慨率估计公式(4.10)及公式(4.11)

**解答：** 

**解答思路：**
1. 贝叶斯估计的一般步骤（详见习题1.1第4步）；
2. 证明公式4.11：假设概率$P_{\lambda}(Y=c_i)$服从狄利克雷（Dirichlet）分布，根据贝叶斯公式，推导后验概率也服从Dirichlet分布，求参数期望；
3. 证明公式4.10：证明同公式4.11。

**解答步骤：**

**第1步：贝叶斯估计的一般步骤**  
参考Wiki：https://en.wikipedia.org/wiki/Bayes_estimator
> 1. 确定参数$\theta$的先验概率$p(\theta)$
> 2. 根据样本集$D={x_1,x_2,\ldots,x_n}$，计算似然函数$P(D|\theta)$：$\displaystyle P(D|\theta)=\prod_{i=1}^n P(x_n|D)$
> 3. 利用贝叶斯公式，求$\theta$的后验概率：$\displaystyle P(\theta|D)=\frac{P(D|\theta)P(\theta)}{\displaystyle \int \limits_\Theta P(D|\theta) P(\theta) d \theta}$ 
> 4. 计算后验概率分布参数$\theta$的期望，并求出贝叶斯估计值：$\displaystyle \hat{\theta}=\int \limits_{\Theta} \theta \cdot P(\theta|D) d \theta$

**第2步：证明公式(4.11)**
$$
\displaystyle P_\lambda(Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k) + \lambda}{N+K \lambda}, \quad k= 1,2,\ldots,K
$$  

证明思路：
1. 条件假设：$P_\lambda(Y=c_k)=u_k$，且服从参数为$\lambda$的Dirichlet分布；随机变量$Y$出现$y=c_k$的次数为$m_k$； 
2. 得到$u$的先验概率$P(u)$；
3. 得到似然函数$P(m|u)$；
4. 根据贝叶斯公式，计算后验概率$P(u|m)$
5. 计算$u$的期望$E(u)$

证明步骤：
1. 条件假设  

&emsp;&emsp;根据朴素贝叶斯法的基本方法，训练数据集$T=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}$，假设：  
（1）随机变量$Y$出现$y=c_k$的次数为$m_k$，即$\displaystyle m_k=\sum_{i=1}^N I(y_i=c_k)$，可知$\displaystyle \sum_{k=1}^K m_k = N$（$y$总共有$N$个）；  
（2）$P_\lambda(Y=c_k)=u_k$，随机变量$u_k$服从参数为$\lambda$的Dirichlet分布。

> **补充说明：**  
> 1. 狄利克雷(Dirichlet)分布  
&emsp;&emsp;参考PRML（Pattern Recognition and Machine Learning）一书的第2.2.1章节：⽤似然函数(2.34)乘以先验(2.38)，我们得到了参数$\{u_k\}$的后验分布，形式为
> $$
p(u|D,\alpha) \propto p(D|u)p(u|\alpha) \propto \prod_{k=1}^K u_k^{\alpha_k+m_k-1}
$$
> &emsp;&emsp;该书中第B.4章节：
狄利克雷分布是$K$个随机变量$0 \leqslant u_k \leqslant 1$的多变量分布，其中$k=1,2,\ldots,K$，并满足以下约束
> $$
0 \leqslant u_k \leqslant 1, \quad \sum_{k=1}^K u_k = 1
$$
记$u=(u_1,\ldots,u_K)^T, \alpha=(\alpha_1,\ldots,\alpha_K)^T$，有
> $$
Dir(u|\alpha) = C(\alpha) \prod_{k-1}^K u_k^{\alpha_k - 1} \\
E(u_k) = \frac{\alpha_k}{\displaystyle \sum_{k=1}^K \alpha_k}
$$
> 2. 为什么假设$Y=c_k$的概率服从Dirichlet分布？  
答：原因如下：  
（1）首先，根据PRML第B.4章节，Dirichlet分布是Beta分布的推广。  
（2）由于，Beta分布是二项式分布的共轭分布，Dirichlet分布是多项式分布的共轭分布。Dirichlet分布可以看作是“分布的分布”；  
（3）又因为，Beta分布与Dirichlet分布都是先验共轭的，意味着先验概率和后验概率属于同一个分布。当假设为Beta分布或者Dirichlet分布时，通过获得大量的观测数据，进行数据分布的调整，使得计算出来的概率越来越接近真实值。  
（4）因此，对于一个概率未知的事件，Beta分布或Dirichlet分布能作为表示该事件发生的概率的概率分布。

2. 得到先验概率  
&emsp;&emsp;根据假设(2)和Dirichlet分布的定义，可得先验概率为
$$
\displaystyle P(u)=P(u_1,u_2,\ldots,u_K) = C(\lambda) \prod_{k=1}^K u_k^{\lambda - 1}
$$

3. 得到似然函数  
&emsp;&emsp;记$m=(m_1, m_2, \ldots, m_K)^T$，可得似然函数为
$$
P(m|u) = u_1^{m_1} \cdot u_2^{m_2} \cdots u_K^{m_K} = \prod_{k=1}^K u_k^{m_k}
$$

4. 得到后验概率分布  
&emsp;&emsp;结合贝叶斯公式，求$u$的后验概率分布，可得
$$
P(u|m) = \frac{P(m|u)P(u)}{P(m)}
$$
&emsp;&emsp;根据假设(1)，可得
$$
P(u|m,\lambda) \propto P(m|u)P(u|\lambda) \propto \prod_{k=1}^K u_k^{\lambda+m_k-1}
$$
&emsp;&emsp;上式表明，后验概率分布$P(u|m,\lambda)$也服从Dirichlet分布

5. 得到随机变量$u$的期望  
&emsp;&emsp;根据后验概率分布$P(u|m,\lambda)$和假设(1)，求随机变量$u$的期望，可得
$$
E(u_k) = \frac{\alpha_k}{\displaystyle \sum_{k=1}^K \alpha_k}
$$
其中$\alpha_k = \lambda+m_k$，则
$$
\begin{aligned}
E(u_k) &= \frac{\alpha_k}{\displaystyle \sum_{k=1}^K \alpha_k} \\
&= \frac{\lambda+m_k}{\displaystyle \sum_{k=1}^K (\lambda + m_k)} \\
&= \frac{\lambda+m_k}{\displaystyle \sum_{k=1}^K \lambda +\sum_{k=1}^K m_k} \quad(\because \sum_{k=1}^K m_k = N) \\
&= \frac{\lambda+m_k}{\displaystyle K \lambda + N } \quad (\because m_k=\sum_{i=1}^N I(y_i=c_k)) \\
&= \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k) + \lambda}{N+K \lambda}
\end{aligned}
$$

&emsp;&emsp;随机变量$u_k$取$u_k$的期望，可得
$\displaystyle P_\lambda(Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k) + \lambda}{N+K \lambda}$，公式(4.11)得证

**第3步：证明公式(4.10)**：
$$
\displaystyle P_{\lambda}(X^{(j)}=a_{jl} | Y = c_k) = \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k) + \lambda}{\displaystyle \sum_{i=1}^N I(y_i=c_k) + S_j \lambda}
$$

证明思路：
1. 条件假设：$P_{\lambda}(X^{(j)}=a_{jl} | Y = c_k)=u_l$，其中$l=1,2,\ldots,S_j$，且服从参数为$\lambda$的Dirichlet分布；出现$x^{(j)}=a_{jl}, y=c_k$的次数为$m_l$； 
2. 得到$u$的先验概率$P(u)$；
3. 得到似然函数$P(m|u)$；
4. 根据贝叶斯公式，计算后验概率$P(u|m)$
5. 计算$u$的期望$E(u)$

证明步骤：
1. 条件假设  

&emsp;&emsp;根据朴素贝叶斯法的基本方法，训练数据集$T=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}$，假设：  
（1）出现$x^{(j)}=a_{jl}, y=c_k$的次数为$m_l$，即$\displaystyle m_l=\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)$，可知$\displaystyle \sum_{l=1}^{S_j} m_l = \sum_{i=1}^N I(y_i=c_k)$（总共有$\displaystyle \sum_{i=1}^N I(y_i=c_k)$个）；  
（2）$P_{\lambda}(X^{(j)}=a_{jl} | Y = c_k)=u_l$，随机变量$u_l$服从参数为$\lambda$的Dirichlet分布。

2. 得到先验概率  
&emsp;&emsp;根据假设(2)和Dirichlet分布的定义，可得先验概率为
$$
\displaystyle P(u)=P(u_1,u_2,\ldots,u_{S_j}) = C(\lambda) \prod_{l=1}^{S_j} u_l^{\lambda - 1}
$$

3. 得到似然函数  
&emsp;&emsp;记$m=(m_1, m_2, \ldots, m_{S_j})^T$，可得似然函数为
$$
P(m|u) = u_1^{m_1} \cdot u_2^{m_2} \cdots u_{S_j}^{m_{S_j}} = \prod_{l=1}^{S_j} u_l^{m_l}
$$

4. 得到后验概率分布  
&emsp;&emsp;结合贝叶斯公式，求$u$的后验概率分布，可得
$$
P(u|m) = \frac{P(m|u)P(u)}{P(m)}
$$
&emsp;&emsp;根据假设(1)，可得
$$
P(u|m,\lambda) \propto P(m|u)P(u|\lambda) \propto \prod_{l=1}^{S_j} u_l^{\lambda+m_l-1}
$$
&emsp;&emsp;上式表明，后验概率分布$P(u|m,\lambda)$也服从Dirichlet分布

5. 得到随机变量$u$的期望  
&emsp;&emsp;根据后验概率分布$P(u|m,\lambda)$和假设(1)，求随机变量$u$的期望，可得
$$
E(u_k) = \frac{\alpha_l}{\displaystyle \sum_{l=1}^{S_j} \alpha_l}
$$
其中$\alpha_l = \lambda+m_l$，则
$$
\begin{aligned}
E(u_l) &= \frac{\alpha_l}{\displaystyle \sum_{l=1}^{S_j} \alpha_l} \\
&= \frac{\lambda+m_l}{\displaystyle \sum_{l=1}^{S_j} (\lambda + m_l)} \\
&= \frac{\lambda+m_l}{\displaystyle \sum_{l=1}^{S_j} \lambda +\sum_{l=1}^{S_j} m_l} \quad(\because \sum_{l=1}^{S_j} m_l = \sum_{i=1}^N I(y_i=c_k)) \\
&= \frac{\lambda+m_l}{\displaystyle S_j \lambda + \sum_{i=1}^N I(y_i=c_k) } \quad (\because m_l=\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)) \\
&= \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k) + \lambda}{\displaystyle \sum_{i=1}^N I(y_i=c_k) + S_j \lambda}
\end{aligned}
$$

&emsp;&emsp;随机变量$u_k$取$u_k$的期望，可得
$\displaystyle P_{\lambda}(X^{(j)}=a_{jl} | Y = c_k) = \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k) + \lambda}{\displaystyle \sum_{i=1}^N I(y_i=c_k) + S_j \lambda}$，公式(4.10)得证。
