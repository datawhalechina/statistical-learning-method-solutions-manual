# 统计学习方法（第二版）习题解答
&emsp;&emsp;李航老师的《统计学习方法》是机器学习领域的经典入门教材之一。  
&emsp;&emsp;本书分为监督学习和无监督学习两篇，全面系统地介绍了统计学习的主要方法。包括感知机、k近邻法、朴素贝叶斯法、决策树、逻辑斯谛回归与最大熵模型、支持向量机、提升方法、EM算法、隐马尔可夫模型和条件随机场，以及聚类方法、奇异值分解、主成分分析、潜在语义分析、概率潜在语义分析、马尔科夫链蒙特卡罗法、潜在狄利克雷分配和PageRank算法等。本书通过第1章、第12章、第13章、第22章进行总结，其余每章都介绍一种统计学习方法。叙述从具体问题或实例入手，由浅入深，阐明思路，给出必要的数学推导，便于读者掌握统计学习方法的实质，学会运用。为满足读者进一步学习的需要，书中还介绍了一些相关研究，给出了少量习题，列出了主要参考文献。  

## 使用说明
&emsp;&emsp;统计学习方法习题解答，主要完成了该书的前11章习题，并提供代码和运行之后的截图，里面的内容是以统计学习方法的内容为前置知识，该习题解答的最佳使用方法是以李航老师的《统计学习方法》为主线，并尝试完成课后习题，如果遇到不会的，再来查阅习题解答。  
&emsp;&emsp;如果觉得解答不详细，可以[点击这里](https://github.com/datawhalechina/statistical-learning-method-solutions-manual/issues)提交你希望补充推导或者习题编号，我们看到后会尽快进行补充。

## 选用的《统计学习方法》版本
<img src="images/statistical-learning-method-book.png?raw=true" width="336" height= "500">


> 书名：统计学习方法  
> 作者：李航  
> 出版社：清华大学出版社  
> 版次：2019年5月第2版  
> 勘误表：http://blog.sina.com.cn/s/blog_7ad48fee0102yjdu.html

## 参考文献
1. [李航《统计学习方法》习题笔记](https://sine-x.com/statistical-learning-method)
2. [李航《统计学习方法笔记》中的代码、notebook、参考文献、Errata](https://github.com/SmirkCao/Lihang)  
3. [CART剪枝详解](https://blog.csdn.net/wjc1182511338/article/details/76793164)
4. [CART剪枝算法详解](http://www.pianshen.com/article/1752163397/)

## 主要贡献者
[@胡锐锋-天国之影-Relph](https://github.com/Relph1119)  
[@王维嘉-king-vega](https://github.com/king-vega)  
[@王茸茸-CiCi-futurewq](https://blog.csdn.net/wangrongrongwq?spm=1000.21115.3001.5343)  
[@王瀚翀-MSQ-ERIC](https://github.com/ericwang970322)

## 项目结构
<pre>
codes----------------------------------------------习题代码
|   +---ch02---------------------------------------第2章习题解答代码
|   |   +---perceptron.py--------------------------习题2.2（构建从训练数据求解感知机模型的例子）
|   +---ch03---------------------------------------第3章习题解答代码
|   |   +---k_neighbors_classifier.py--------------习题3.1（k近邻算法关于k值的模型比较）
|   |   +---kd_tree_demo.py------------------------习题3.2（kd树的构建与求最近邻点）
|   |   +---my_kd_tree.py--------------------------习题3.3（用kd树的k邻近搜索算法）
|   +---ch05---------------------------------------第5章习题解答代码
|   |   +---k_neighbors_classifier.py--------------习题5.1（调用sklearn的DecisionTreeClassifier类使用C4.5算法生成决策树）
|   |   +---my_decision_tree.py--------------------习题5.1（自编程实现C4.5生成算法）
|   |   +---my_least_squares_regression_tree.py----习题5.2（最小二乘回归树生成算法）
|   +---ch06---------------------------------------第6章习题解答代码
|   |   +---my_logistic_regression.py--------------习题6.2（实现Logistic回归模型学习的梯度下降法）
|   |   +---maxent_dfp.py--------------------------习题6.3（最大熵模型学习的DFP算法）
|   +---ch07---------------------------------------第7章习题解答代码
|   |   +---svm_demo.py----------------------------习题7.2（根据题目中的数据训练SVM模型，并在图中画出分离超平面、间隔边界及支持向量）
|   +---ch08---------------------------------------第8章习题解答代码
|   |   +---adaboost_demo.py-----------------------习题8.1（使用sklearn的AdaBoostClassifier分类器实现）
|   |   +---my_adaboost.py-------------------------习题8.1（自编程实现AdaBoost算法）
|   +---ch09---------------------------------------第9章习题解答代码
|   |   +---three_coin_EM.py-----------------------习题9.1（三硬币模型的EM算法）
|   |   +---gmm_demo.py----------------------------习题9.3（使用GaussianMixture求解两个分量高斯混合模型的6个参数）
|   |   +---my_gmm.py------------------------------习题9.3（自编程实现求两个分量的高斯混合模型的5个参数）
|   +---ch10---------------------------------------第10章习题解答代码
|   |   +---hidden_markov_backward.py--------------习题10.1（隐马尔可夫模型的后向算法）
|   |   +---hidden_markov_forward_backward.py------习题10.2（隐马尔可夫模型的前向后向算法）
|   |   +---hidden_markov_viterbi.py---------------习题10.3（隐马尔可夫模型的维特比算法）
|   +---ch11---------------------------------------第11章习题解答代码
|   |   +---crf_matrix.py--------------------------习题11.4（使用条件随机场矩阵形式，计算所有路径状态序列的概率及概率最大的状态序列）
docs---------------------------------------习题解答
notebook-----------------------------------习题解答JupyterNotebook格式
requirements.txt---------------------------运行环境依赖包
</pre>

## 关注我们
<div align=center><img src="images/qrcode.jpeg" width = "250" height = "270" alt="Datawhale，一个专注于AI领域的学习圈子。初衷是for the learner，和学习者一起成长。目前加入学习社群的人数已经数千人，组织了机器学习，深度学习，数据分析，数据挖掘，爬虫，编程，统计学，Mysql，数据竞赛等多个领域的内容学习，微信搜索公众号Datawhale可以加入我们。"></div>

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。