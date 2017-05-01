正则化总结
(需要引用我之前的文章：#机器学习的本质)

生成模型、判别模型

VC—DIM、自由度

L1-norm and L2-norm
Sparse -> generalization
Feature Selection

cross-validation, early stopping, pruning, dropout, Bayesian priors on parameters or model comparison

ICLR 2017 Best Paper Award 
Generalization vs. Memory

data-dependent regularization



行列式不为零的方阵，才有对应的逆矩阵


#经验风险=平均损失函数
结构风险=损失函数+正则化项（惩罚项）正则化是结构风险最小化的策略。正则化项一般是模型复杂度的单调递增函数，模型越复杂，正则化值就越大。比如，模型参数向量的范数。

#正则化项的引入其实是利用了先验知识
体现了人对问题的解的认知程度或者对解的估计。这样就可以将人对该问题的理解和需求（先验知识）融入到模型的学习当中，对模型参数设置先验，强行地让学习到的模型具有人想要的特性，例如稀疏、低秩、平滑等等。（正则与稀疏、低秩和平滑的关系）L1正则是laplace先验，l2是高斯先验，分别由参数sigma确定


#Reference
-关于L0，L1和L2范数的规则化 的比较全的blog: http://blog.csdn.net/fightsong/article/details/53311582


本质上，是加入先验、规则，缩小解空间


# Bias and Variance Decomposition
$$E(y-f(x))^2=E(y-\tilde{f}(x)+\tilde{f}(x)-f(x))^2=E(y-\tilde{f}(x))^2+E(f(x)-\tilde{f}(x))^2$$

# VC-Dimension

$$Pr(test\ error\leq training\ error+\sqrt{\frac{1}{N}[D(log(\frac{2N}{D}+1)-log(\frac{\eta}{4}))]})=1-\eta$$


# Linear Algebra
## Define the Problem
$$X^{m,n}=
\begin{bmatrix}
x_{11}&x_{12}&...&x_{1n}\\
x_{21}&x_{22}&...&x_{2n}\\
...&...&...&...\\
x_{m1}&x_{m2}&...&x_{mn}\\
\end{bmatrix}$$

$$y=
\begin{bmatrix}
y_{1}&y_{2}&...&y_{n}
\end{bmatrix}^T$$

$$X^Tw=y$$

## Least Square
$$E(w|X,y)=\sum_{i=1}^n\epsilon_i^2=\sum_{i=1}^n(x_i^Tw-y_i)^2=||X^Tw-y||^2$$
$$\nabla E(w|X,y)=\nabla ||X^Tw-y||^2=2(XX^Tw-Xy)=0$$
$$w=(XX^T)^{-1}Xy=X^+y$$

## weight decay
$$E(w|X,y)=||X^Tw-y||^2+\nu||w||^2$$
$$w=(XX^T+\nu I)^{-1}Xy$$
$$\frac{|\lambda_{max}|}{|\lambda_{min}|}\rightarrow \frac{|\lambda_{max}+\nu|}{|\lambda_{min}+\nu|}<\frac{|\lambda_{max}|}{|\lambda_{min}|}$$

## Condition Number

$$V=\begin{bmatrix}v_1&v_1&...&v_m\end{bmatrix}$$
$$\Lambda=\begin{bmatrix}
\lambda_1&0 &...&0\\
0&\lambda_2 &...&0\\
...&...&...&...\\
0&0&...&\lambda_m \\
\end{bmatrix}$$
$$M=MVV^T=V\Lambda V^T$$

$$M^{-1}=V\Lambda^{-1}V^T$$

$$\Lambda^{-1}=\begin{bmatrix}
\frac{1}{\lambda_1}&0 &...&0\\
0&\frac{1}{\lambda_2} &...&0\\
...&...&...&...\\
0&0&...&\frac{1}{\lambda_m} \\
\end{bmatrix}$$
$$\frac{||M||_H}{||M^{-1}||_H}=\frac{|\lambda_{max}|}{|\lambda_{min}|}$$

$$w=(XX^T+\nu I)^{-1}Xy=(V\Lambda V^{-1}+V(\nu I)V^{-1})^{-1}Xy=V(\Lambda+\nu I)^{-1}V^{-1}Xy$$

### 正定二次型(positive definite quadratic form) 以及 正定矩阵(positive definite matrix)
####定义：
$\forall x(非0)$ 恒有 $f(x)=x^TAx>0$，则称f为正定二次型，A为正定矩阵。  
例如：
设 $f(x_1,x_2,x_3)=x_1^2+3x_2^2+x_3^2$，则  
$\forall x=\begin{bmatrix}t\\\\u\\\\v\end{bmatrix}\neq 0$，恒有 $f(t,u,v)=t^2+3u^2+v^2>0$, 所以f正定，$A=\begin{bmatrix}1&&\\\\&3&\\\\&&1\end{bmatrix}$为正定矩阵  

####半正定矩阵
把上述正定矩阵中的大于零替换为大于或等于零，就是半正定矩阵

由于 $XX^T$ 是半正定矩阵($positive semi-definite matrix$)所有 $\lambda$都是

$$v^TMv=v^TXX^Tv=u^Tu=||u||^2\ge0\ (u=X^Tv)$$
所以说我们关注的矩阵$M$是半正定矩阵，根据其性质，所有特征值都大于等于0

## example
$$
\begin{bmatrix}1&2\\1&1.999\end{bmatrix}
\begin{bmatrix}w_0\\w_1\end{bmatrix}=
\begin{bmatrix}4.000\\3.999\end{bmatrix}
\rightarrow
w=\begin{bmatrix}2.0000\\1.0000\end{bmatrix},
eigenbasis=\begin{bmatrix}0.0002\\-5.6562\end{bmatrix}
$$

$$
\begin{bmatrix}1&2\\1&1.999\end{bmatrix}
\begin{bmatrix}w_0\\w_1\end{bmatrix}=
\begin{bmatrix}4.001\\3.998\end{bmatrix}
\rightarrow
w=\begin{bmatrix}-1.999\\3.000\end{bmatrix},
eigenbasis=\begin{bmatrix}-0.0012\\-5.6558\end{bmatrix}
$$

After adding $0.1I$ regularization term
$$
\begin{bmatrix}1&2\\1&1.999\end{bmatrix}
\begin{bmatrix}w_0\\w_1\end{bmatrix}=
\begin{bmatrix}4.000\\3.999\end{bmatrix}
\rightarrow
w=\begin{bmatrix}1.2884\\1.2914\end{bmatrix},
eigenbasis=\begin{bmatrix}0.0002\\-5.6562\end{bmatrix}
$$

$$
\begin{bmatrix}1&2\\1&1.999\end{bmatrix}
\begin{bmatrix}w_0\\w_1\end{bmatrix}=
\begin{bmatrix}4.001\\3.998\end{bmatrix}
\rightarrow
w=\begin{bmatrix}1.3017\\1.2846\end{bmatrix},
eigenbasis=\begin{bmatrix}-0.0012\\-5.6558\end{bmatrix}
$$



# Semi-supervised learning









