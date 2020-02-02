# 第4章朴素贝叶斯法-习题

## 习题4.1
&emsp;&emsp;用极大似然估计法推出朴素贝叶斯法中的概率估计公式(4.8)及公式 (4.9)。

**解答：**  
**第1步：**证明公式(4.8)：$\displaystyle P(Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k)}{N}$  
由于朴素贝叶斯法假设$Y$是定义在输出空间$\mathcal{Y}$上的随机变量，因此可以定义$P(Y=c_k)$概率为$p$。  
令$\displaystyle m=\sum_{i=1}^NI(y_i=c_k)$，得出似然函数：$$L(p)=f_D(y_1,y_2,\cdots,y_n|\theta)=\binom{N}{m}p^m(1-p)^{(N-m)}$$使用微分求极值，两边同时对$p$求微分：$$\begin{aligned}
0 &= \binom{N}{m}\left[mp^{(m-1)}(1-p)^{(N-m)}-(N-m)p^m(1-p)^{(N-m-1)}\right] \\
& = \binom{N}{m}\left[p^{(m-1)}(1-p)^{(N-m-1)}(m-Np)\right]
\end{aligned}$$可求解得到$\displaystyle p=0,p=1,p=\frac{m}{N}$  
显然$\displaystyle P(Y=c_k)=p=\frac{m}{N}=\frac{\displaystyle \sum_{i=1}^N I(y_i=c_k)}{N}$，公式(4.8)得证。

----

**第2步：**证明公式(4.9)：$\displaystyle P(X^{(j)}=a_{jl}|Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)}{\displaystyle \sum_{i=1}^N I(y_i=c_k)}$  
令$P(X^{(j)}=a_{jl}|Y=c_k)=p$，令$\displaystyle m=\sum_{i=1}^N I(y_i=c_k), q=\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)$，得出似然函数：$$L(p)=\binom{m}{q}p^q(i-p)^{m-q}$$使用微分求极值，两边同时对$p$求微分：$$\begin{aligned}
0 &= \binom{m}{q}\left[qp^{(q-1)}(1-p)^{(m-q)}-(m-q)p^q(1-p)^{(m-q-1)}\right] \\
& = \binom{m}{q}\left[p^{(q-1)}(1-p)^{(m-q-1)}(q-mp)\right]
\end{aligned}$$可求解得到$\displaystyle p=0,p=1,p=\frac{q}{m}$  
显然$\displaystyle P(X^{(j)}=a_{jl}|Y=c_k)=p=\frac{q}{m}=\frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)}{\displaystyle \sum_{i=1}^N I(y_i=c_k)}$，公式(4.9)得证。

## 习题4.2
&emsp;&emsp;用贝叶斯估计法推出朴素贝叶斯法中的慨率估计公式(4.10)及公式(4.11)

**解答：** 
**第1步：**证明公式(4.11)：$\displaystyle P(Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k) + \lambda}{N+K \lambda}$  
假设$P_\lambda(Y=c_i)=\theta_i,i=1,2,\cdots,K$是随机变量，且$\theta_i,i=1,2,\cdots,K$的先验分布是服从参数为$\lambda$的Dirichlet分布：
$$P(\theta_1,\theta_2,\cdots,\theta_K|\lambda)=\frac{1}{B(\lambda)}\prod_{i=1}^{K}\theta_i^{\lambda-1}\quad(1)$$
考虑训练数据集$T={(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)}$，记$\displaystyle M_i=\sum_{j=1}^NI(y_i=c_i)$，其中$i=1,2,\cdots-K$为随机变量。  
令$\theta=(\theta_1,\theta_2,\cdots,\theta_K),M=(M_1,M_2,\cdots,M_K)$，用上述数据集改进先验分布(1)，得到后验分布：
$$P(\theta|M)=\frac{P(M|\theta) \cdot P(\theta)}{\int P(M|\theta)P(\theta)d\theta}\quad(2)$$
其中(2)式的分母$\int P(M|\theta)P(\theta)d\theta$是一个定值，与$\theta$无关。假设$P(M|\theta)$服从多项分布：
$$P(M|\theta)=\theta_1^{M_1}\cdot\theta_2^{M_2}\cdots\theta_K^{M_K}\quad(3)$$
将(1),(3)式代入(2)式中，可得：
$$P(\theta|M) \propto \prod_{i=1}^{K}\theta_i^{\lambda+M_i-1} $$
由上式可知，后验概率$P(\theta|M)$也服从Dirichlet分布，因此$P_\lambda(Y=c_i)$取随机变量$\theta_i$的期望，即
$$P(Y=c_k) = E(\theta) = \frac{M_i+\lambda}{N+K\lambda}=\frac{\displaystyle \sum_{i=1}^N I(y_i=c_k) + \lambda}{N+K \lambda}$$
公式(4.11)得证。 

----

**第2步：**证明公式(4.10)：$\displaystyle P_{\lambda}(X^{(j)}=a_{jl} | Y = c_k) = \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k) + \lambda}{\displaystyle \sum_{i=1}^N I(y_i=c_k) + S_j \lambda}$   
根据第1步，可同理得到$$
P(Y=c_k, x^{(j)}=a_{j l})=\frac{\displaystyle \sum_{i=1}^N I(y_i=c_k, x_i^{(j)}=a_{jl})+\lambda}{N+K S_j \lambda}$$  
$$\begin{aligned} 
P(x^{(j)}=a_{jl} | Y=c_k)
&= \frac{P(Y=c_k, x^{(j)}=a_{j l})}{P(y_i=c_k)} \\
&= \frac{\displaystyle \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k, x_i^{(j)}=a_{jl})+\lambda}{N+K S_j \lambda}}{\displaystyle \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k) + \lambda}{N+K \lambda}} \\
&= (\lambda可以任意取值，于是取\lambda = S_j \lambda) \\
&= \frac{\displaystyle \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k, x_i^{(j)}=a_{jl})+\lambda}{N+K S_j \lambda}}{\displaystyle \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k) + \lambda}{N+K S_j \lambda}} \\ 
&= \frac{\displaystyle \sum_{i=1}^N I(y_i=c_k, x_i^{(j)}=a_{jl})+\lambda}{\displaystyle \sum_{i=1}^N I(y_i=c_k) + \lambda} (其中\lambda = S_j \lambda)\\
&= \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k) + \lambda}{\displaystyle \sum_{i=1}^N I(y_i=c_k) + S_j \lambda}
\end{aligned} $$  
公式(4.11)得证。
