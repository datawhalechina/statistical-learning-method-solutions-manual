# 第1章统计学习方法概论

## 习题1.1
&emsp;&emsp;说明伯努利模型的极大似然估计以及贝叶斯估计中的统计学习方法三要素。伯努利模型是定义在取值为0与1的随机变量上的概率分布。假设观测到伯努利模型$n$次独立的数据生成结果，其中$k$次的结果为1，这时可以用极大似然估计或贝叶斯估计来估计结果为1的概率。

**解答：**  

**解答思路：**
1. 写出伯努利模型的概率分布函数；
2. 写出伯努利模型的极大似然估计以及贝叶斯估计中的统计学习方法三要素；
3. 根据伯努利模型的极大似然估计算法，估计结果为1的概率；
4. 根据伯努利模型的贝叶斯估计算法，估计结果为1的概率。

**解答步骤：**  

**第1步：伯努利模型的概率分布函数**  
根据题意：伯努利模型是定义在取值为0与1的随机变量上的概率分布。  
对于随机变量$X$，则有：
$$
f(X=1)=p \\ f(X=0)=1-p
$$
由于随机变量$X$只有0和1两个值，$X$的概率分布函数可写为：
$$
f(X=x|p)=p^x (1-p)^{(1-x)} \quad 0<p<1
$$
根据题意，假设观测到伯努利模型$n$次独立的数据生成结果，其中$k$次的结果为1，则概率质量函数可以写成：
$$
f(X|p)=P\{X=k\}=C_n^k p^k (1-p)^{n-k}
$$

**第2步：伯努利模型的极大似然估计以及贝叶斯估计中的统计学习方法三要素**  
（1）极大似然估计  
模型：$\mathcal{F}=\{f|f(x|p)=p^x(1-p)^{(1-x)}\}$  
策略：最大化似然函数  
算法：$\displaystyle \mathop{\arg\max}_{p} L(p|X)= \mathop{\arg\max}_{p} f(X|p)$ 

（2）贝叶斯估计  
模型：$\mathcal{F}=\{f|f_p(x)=p^x(1-p)^{(1-x)}\}$  
策略：求参数期望  
算法：
$$
\begin{aligned}  E_\pi [p |X ]
&= {\int_0^1} p \pi (p|X) dp \\
&= {\int_0^1} p \frac{f(X|p)\pi(p)}{\int_{\Omega}f(X|p)\pi(p)dp}dp \\
\end{aligned}
$$

**第3步：伯努利模型的极大似然估计**  
> 极大似然估计的一般步骤：  
参考Wiki：https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1   
> 1. 写出随机变量的概率分布函数；  
> 2. 写出似然函数；
> 3. 对似然函数取对数，得到对数似然函数，并进行化简；
> 4. 对参数进行求导，并令导数等于0；
> 5. 求解似然函数方程，得到参数的值。

定义$P(X=1)$概率为$p$，可得似然函数为：
$$
\begin{aligned} L(p|X) &= f(X|p) \\
&= C_n^k p^k (1-p)^{n-k}
\end{aligned}
$$
对似然函数取对数，得到对数似然函数为：
$$
\begin{aligned} \log L(p|X) &= \log C_n^k p^k (1-p)^{n-k} \\
&= C_n^k \left[\log(p^k) + \log\left( (1-p)^{n-k} \right)\right] \\
&= C_n^k \left[ k\log p + (n-k)\log (1-p)\right]
\end{aligned}
$$
求解参数$p$：
$$
\begin{aligned}
\hat{p} &= \mathop{\arg\max}_p L(p|X) \\
&= \mathop{\arg\max}_p C_n^k \left[ k\log p + (n-k)\log (1-p)\right]
\end{aligned}
$$
对参数$p$求导，并求解导数为0时的$p$值：
$$
\begin{aligned}
\frac{\partial \log L(p)}{\partial p} &= C_n^k \left[ \frac{k}{p} - \frac{n-k}{1-p} \right] \\
&= C_n^k \left[ \frac{k(1-p) - p(n-k)}{p(1-p)} \right] \\
&= C_n^k \frac{k-np}{p(1-p)} = 0
\end{aligned}
$$
从上式可得，$k-np=0$，即$\displaystyle P(X=1)=p=\frac{k}{n}$

**第4步：伯努利模型的贝叶斯估计**  
> 贝叶斯估计的一般步骤：  
参考Wiki：https://en.wikipedia.org/wiki/Bayes_estimator
> 1. 确定参数$\theta$的先验分布$p(\theta)$
> 2. 根据样本集$D={x_1,x_2,\ldots,x_n}$，计算样本的联合分布$p(D|\theta)$：$\displaystyle p(D|\theta)=\prod_{i=1}^n p(x_n|D)$
> 3. 利用贝叶斯公式，求$\theta$的后验分布：$\displaystyle p(\theta|D)=\frac{p(D|\theta)p(\theta)}{\displaystyle \int \limits_\Theta p(D|\theta) p(\theta) d \theta}$ 
> 4. 求出贝叶斯估计值：$\displaystyle \hat{\theta}=\int \limits_{\Theta} \theta p(\theta|D) d \theta$

定义$P(Y=1)$概率为$p$，$p$在$[0,1]$之间的取值是等概率的，因此先验概率密度函数$\pi(p) = 1$，可得似然函数： 
$$
L(p)=f(X|p)=C_n^k p^k (1-p)^{n-k}
$$  
根据似然函数和先验概率密度函数，可以求解$p$的后验分布为：
$$
\begin{aligned}
\pi(p|X) & =\frac{f(X|p)\pi(p)}{\int_{\Omega}f(X|p)\pi(p)dp} \\
&= \frac{C_n^k p^k (1-p)^{n-k}}{\displaystyle \int_0^1 C_n^k p^k (1-p)^{n-k} dp} \\
&= \frac{p^k (1-p)^{n-k}}{\displaystyle \int_0^1 p^k (1-p)^{n-k} dp}
\end{aligned}
$$

> Beta函数：  
来源百度百科：https://baike.baidu.com/item/%E8%B4%9D%E5%A1%94%E5%87%BD%E6%95%B0?fromtitle=Beta%E5%87%BD%E6%95%B0&fromid=50806259  
对任意实数$P,Q>0$，有$$B(P,Q)=\int_0^1 x^{P-1} (1-x)^{Q-1} dx$$称该函数为贝塔函数，或为Beta函数、B函数。  
递推公式：
> 1. $\displaystyle B(P,Q)=\frac{Q-1}{P+Q-1} B(P,Q-1) \quad P>0,Q>1$
> 2. $\displaystyle B(P,Q)=\frac{P-1}{P+Q-1} B(P-1,Q) \quad P>1,Q>0$
> 3. $\displaystyle B(P,Q)=\frac{(P-1)(Q-1)}{(P+Q-1)(P+Q-2)} B(P-1,Q-1) \quad P>1,Q>1$

根据Beta函数定义：$\displaystyle \int_0^1 p^k (1-p)^{n-k} dp=B(k+1,n-k+1)$  
$\therefore p$的后验分布为$\displaystyle \pi(p|X) =\frac{p^k(1-p)^{n-k}}{B(k+1,n-k+1)}$  
则$p$的期望为：
$$
\begin{aligned}
E_\pi[p|X] &= {\int}p\pi(p|X)dp \\
&= {\int_0^1} p \frac{p^k (1-p)^{n-k}}{B(k+1,n-k+1)}dp \\
&= {\int_0^1} \frac{p^{k+1} (1-p)^{n-k}}{B(k+1,n-k+1)}dp \\
&= \frac{\displaystyle \int_0^1 p^{k+1} (1-p)^{n-k} dp}{B(k+1,n-k+1)}
\end{aligned}
$$
根据Beta函数定义：$\displaystyle \int_0^1 p^{k+1} (1-p)^{n-k} dp = B(k+2, n-k+1)$   
$\therefore \displaystyle E_\pi[p|X] = \frac{B(k+2,n-k+1)}{B(k+1,n-k+1)}$  
根据Beta函数的递推公式：
> $\displaystyle B(P,Q)=\frac{P-1}{P+Q-1} B(P-1,Q) \Rightarrow \frac{B(P,Q)}{B(P-1,Q)} = \frac{P-1}{P+Q-1}$   

$\therefore \displaystyle E_\pi[p|X] = \frac{B(k+2,n-k+1)}{B(k+1,n-k+1)} = \frac{k+2-1}{k+2+n-k+1-1} = \frac{k+1}{n+2}$  
$\therefore \displaystyle P(X=1)=\frac{k+1}{n+2}$

## 习题1.2
&emsp;&emsp;通过经验风险最小化推导极大似然估计。证明模型是条件概率分布，当损失函数是对数损失函数时，经验风险最小化等价于极大似然估计。

**解答：**

**解答思路：**  
1. 根据经验风险最小化定义，写出目标函数；
2. 根据对数损失函数，对目标函数进行整理；
3. 根据似然函数定义和极大似然估计的一般步骤（计算时需要取对数），可得到结论。

**解答步骤：**  
假设模型的条件概率分布是$P_{\theta}(Y|X)$，样本集$D=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}$，根据书中第17页，对数损失函数为：
$$
L(Y,P(Y|X)) = -\log P(Y|X)
$$
根据书中第18页，按照经验风险最小化求最优模型就是求解最优化问题：
$$
\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N L(y_i, f(x_i))
$$
结合上述两个式子，可得经验风险最小化函数：
$$
\begin{aligned} 
\mathop{\arg\min}_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N L(y_i, f(x_i)) &= \mathop{\arg\min}_{f \in \mathcal{F}} \frac{1}{N} \sum_D [-\log P(Y|X)] \\
&= \mathop{\arg\max}_{f \in \mathcal{F}} \frac{1}{N} \sum_D \log P(Y|X) \\
&= \mathop{\arg\max}_{f \in \mathcal{F}} \frac{1}{N} \log \prod_D P(Y|X) \\
&= \frac{1}{N} \mathop{\arg\max}_{f \in \mathcal{F}} \log \prod_D P(Y|X)
\end{aligned}
$$
根据似然函数定义：$\displaystyle L(\theta)=\prod_D P_{\theta}(Y|X)$，以及极大似然估计的一般步骤，可得：
$$
\mathop{\arg\min}_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N L(y_i, f(x_i)) = \frac{1}{N} \mathop{\arg\max}_{f \in \mathcal{F}} \log L(\theta)
$$
即经验风险最小化等价于极大似然估计，得证。
