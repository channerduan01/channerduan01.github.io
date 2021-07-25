# Gausian Process 高斯过程


## 1. Stochastic Process 随机过程
高斯过程是随机过程的特例，所以我们先阐述一下随机过程的一般概念。

### 1.1 定义
A collection of random variables indexed by time or space.


A stochastic process can be written as $\{X(t,\omega)\ :\ \ t\in T,\omega \in \Omega\}$ to reflect that it is a function of 2 variables.
随机过程是一个定义在 $T\ \times\ \Omega$ 的二元函数，大多数问题定义下 $t\in T$ 表示时间维度。

- 对于一个特定的 $t=t_1$，$X_{t_1}(\omega)$ 是一个随机变量，包含了所有样本函数在 $t_1$ 时刻的取值。随机过程可以看作是这一组随机变量（$t\in T$）的全体。
- 对于一个特定的 $\omega=\omega_1$，$X_{\omega_1}(t)$ 是一条确定的样本轨迹/样本函数，通过一系列的 $X_{\omega_1}(t),\ t\in T$ 组成，这个过程可以可视化成一条曲线。随机过程也可看作是所有样本函数（$\omega\in \Omega$）的集合，$\Omega$ 被成为样本空间 sample space。


### 1.2 示例
random walk 随机游走 就属于 随机过程。

有很多统计问题示例，详见 [知乎-什么是随机过程？](https://www.zhihu.com/question/280948058)

## 2. Gausian Process 高斯过程
### 2.1 定义
A time continuous stochastic process $\{X_t;t\in T\}$ is Gaussian if and only if for every finite set of indices $t_1,...,t_k$ in the index set $T$ $X_{t_1,...,t_k}=(X_{t_1},...,X_{t_k})$ is a multivariate Gaussian random variable.

GP（Gaussian Process）高斯过程 也属于随机过程，具体有3个额外特点：

- 连续域：观测值出现在一个连续空间中（例如最常见的时间，即 $t\in T$）
- 正态分布：空间中的每一个点（即 $X_{t_1}$），都关联一个相应的正态分布的随机变量
- 多元正态分布：这些随机变量的每个有限集合都有一个多元正态分布

高斯过程 是一个定义在连续域上的无限多个高斯随机变量所组成的随机过程，高斯过程是一个无限维的高斯分布。

The distribution of a Gaussian process is the joint distribution of all those (infinitely many) random variables, and as such, it is a distribution over functions with a continuous domain, e.g. time or space.

### 2.2 示例
我们假设 $t\in T$ 是表示年龄的连续域，而 $X$ 测量的是收入水平（一个整数）。

- 对于任意 $t$，例如 $t=中年$；$X_{t=中年}\ $ 是一个正态分布的**随机变量**，表示整体中年人收入水平分布，具体的分布细节是已经确定的 $X_{t=中年}\ \thicksim N(\mu_{t=中年}\ \ ,\  \sigma_{t=中年}^2\ \ )$
- 对于一个随机变量的有限集合，例如 $t \in \{青年, 中年, 老年\}$ 对应的 $(X_{t=青年}\ \ , X_{t=中年}\ \ , X_{t=老年}\ \ )$ 整体上收入水平服从一个多元正态分布，即 $\thicksim N(\mu,\Sigma)$。这里面不同时间点的收入水平应该有一定关联关系（例如青年时期收入偏高，则中年时期收入也偏高），这就通过协方差矩阵 $\Sigma$ 来体现。
- 额外地（这一项不重要），对于某一个具体的人，我们能画出一条其收入随时间变化的曲线，这就是该样本的函数了。

### 2.3 建模
首先看 多维高斯分布：对于一个 $n$ 维的高斯分布，其通过 $n$ 维均值向量 $\mu_n$ 以及 $n\times n$ 维协方差矩阵 $\Sigma_{n\times n}$。

而对于定义在连续域 $t\in T$ 上 无限维的高斯过程，其需要描述无限个维度的均值，以及无限个维度相互之间的协方差。其通过定义函数来完成：

- $m(t)$ 表示任意一个时刻 $t$ 的均值
- $k(s, t)$ 表示任意两个时刻 $s$ 和 $t$ 之间的 kernel function 核函数，也称 convariance function 协方差函数；即表示两个时刻之间的相关性。核函数确定了高斯过程先验下函数的潜在结构

这就是高斯过程的两个核心要素：均值函数 & 核函数，一旦这两个要素确定了，整个高斯过程就确定了 $GP(m(t),k(t,s))$。

这里有个系统性的回答很不错：[知乎-如何通俗易懂地介绍 Gaussian Process？-石溪](https://www.zhihu.com/question/46631426/answer/102211375)

### 2.4 常见核函数
#### RBF（Radius Basis Function）径向基函数
定义公式如下：
$k(s,t)=\sigma^2exp(-\large\frac{||s-t||^2}{2l^2})$
其中 $\sigma$ 和 $l$ 是超参数，例如 $\sigma=1,l=1$；而 $||s-t||^2$ 直接衡量了两个时刻之间的距离。通过$exp(-x)$单调递减函数，则两个时刻距离越远，协方差越小，相关性越小，直到无限接近0；反之则两个时刻举例越近，协方差越大，相关性越大，最多到 $\sigma^2=1$。







