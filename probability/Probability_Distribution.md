# Bernoulli、Binomial、Beta 分布详解
本文关注 离散随机变量 discrete random variable 相关的分布；相对的，连续随机变量 continuous random variable 如经典的 高斯分布（Gaussian Distribution）会在其他文章中介绍。

典型的离散随机变量的分布有：

- 伯努利分布（Bernoulli Distribution）
- 二项分布（Binomial Distribution）
- 分类分布（Categorical Distribution）
- 多项分布（Multinomial Distribution）

以及他们的共轭先验分布：

- 贝塔分布（Beta Distribution）
- 狄利克雷分布（Dirichelet Distribution）

貌似很多分布，但都以简单的 Bernoulli 为基础建立，非常规律，而且很多性质是相同的。本文将从 Bernoulli 展开介绍各个分布，给出最大似然的推导应用，最后再介绍 conjugacy 的概念。

## 1. 各分布详解
### 1.1 伯努利分布（又称 01分布）
这是最基本的离散随机变量的概率分布，相当于统计理论的 Hello World，我们记一个变量 $y\in \{1,\ 0\}$，即某个事件或者试验只有两种结果，1或0，可以理解为发生或未发生；例如经典的抛硬币试验，我们只能得到俩种结果，heads or tails。

我们通常引入 $\mu \in [0, 1]$ 的实数表示得到结果 $y=1$ 的概率（例如硬币正面朝上）：
$$p(y=1|\mu)=\mu,\ y\in \{1,\ 0\}$$
这一类分布称为 伯努利分布：
$$Bern(y|\mu)=\mu^y(1-\mu)^{1-y}$$
没有比这更简单的分布了，仅定义一个参数，有两种可能结果。但是在机器学习中 Bern 却极其常用，例如 Logistic 函数值实际就是对 Bernoulli Distribution 中参数 $\mu$ 的估计。

### 1.2 伯努利分布 与 二项分布、分类分布、多项分布
总的来说，伯努利分布 是基础；二项分布 推广到N次试验，分类分布 推广到单次试验K种结果，多项分布 推广到N次试验且每次K种结果。

- Bernoulli：**单次随机试验，只有两种可能的结果；**也可以称为N=1时的二项分布。$$Bern(y|\mu)=\mu^y(1-\mu)^{1-y}$$
- Binomial：**N次独立 Bernoulli 试验，得到成功次数结果的离散分布；**例如5次抛硬币，正面朝上的次数（这时有6种可能）的离散概率分布。$$Bin(m|N,\mu)=\frac{N!}{m!(N-m)!}\mu^m(1-\mu)^{(N-m)}$$
- Categorical：**对 Bernoulli 的推广，也是单次随机试验，有K种可能的结果（互斥）**，很少有用，公式和 Multinomial 类似（去掉阶乘部分），不写了。
- Multinomial：**N次独立试验，每次试验有K种可能的结果（互斥）**，每次试验也被称为 Categorical，可以说是最 general 的分布。最常用的例子是扔骰子。$$Mult(m_1,m_2,...\ ,m_K|\mu_1,\mu_2,...\ ,\mu_K,N)=\frac{N!}{m_1!m_2!\ ...\ m_K!}\prod_{k=1}^K{\mu_k^{m_k}}\ \ \ \ s.t.\sum_k^K{\mu_k}=1$$

上述公式都非常简洁，还可以更简洁：

- 繁琐的阶乘计算是用于列举出特定次数时的组合数（例如N次抛硬币m向上一共有多少种组合），如果假设只有N=1次实验的话可以直接消去（0!=1!=1）。
- Bernoulli 试验结果只有俩种可能，所以使用一个 $\mu$ 足够，以 $1-\mu$ 标识另一种情况的概率。而 Multinomial 则显示地指定了所有$\mu_k$，其实这些分布都可以用 Multinomial 的形式表示。

## 2. MLE 最大似然估计 与 各个分布

### 2.1 Bernoulli (Binomial N=1)
假设我们得到了对于二元（0或1）随机变量 $y$ 的观察结果 $D=\{y_1,y_2,...\ ,y_N\}$，假设 $y \thicksim Bern(\mu)$ 或者 $Bio(\mu, 1)$，那我们如何估计参数 $\mu$ 呢？以下标准的三步法 最大似然估计 MLE：

- 写出似然函数：$$L=p(D|\mu)=\prod_{n=1}^N{p(y_n|\mu)}=\prod_{n=1}^N{\mu^{y_n}(1-\mu)^{1-y_n}}$$
- 似然函数取对数化简整理：$$l=ln\ p(D|\mu)=\sum_{n=1}^N{ln\{\mu^{y_n}(1-\mu)^{1-y_n}\}}=\sum_{n=1}^N\{y_nln\mu+(1-y_n)ln(1-\mu)\}$$
- 最大化似然函数，取导数为0的极值：$$\frac{\partial\ l}{\partial\ \mu}=\sum_{n=1}^N\{y_n\frac{1}{\mu}-(1-y_n)\frac{1}{1-\mu}\}=\frac{1}{\mu(1-\mu)}\sum_{n=1}^N\{y_n-\mu\}=0$$ $$\mu_{MLE}=\frac{1}{N}\sum_{n=1}^Ny_n,\ \mu_{MLE}=\displaystyle\frac{m}{N}$$

上式中 $m$ 定义为这 N 次试验中，$y=1$ 的次数。MLE 的估计结果非常符合直观，例如我们想了解抛一枚硬币结果朝上的概率，就进行 $N$ 试验，以观察到朝上的次数 $m$ 直接除以 $N$ 得到这一概率。

这里还有一个很有意思的结果，似然函数取对数的结果就是 Logistic Regression 的 loss 函数形式，因为 Logistic Regression 本身就基于 Bernoulli，如同 Linear Regression 基于 Gaussian Distribution。后面会专门以 GLM（General Linear Model）为话题在其他文章中深入讨论。

### 2.2 Binomial
随机变量 $y$ 的观察结果 $D=\{y_1,y_2,...\ ,y_N\}$，假设 $y \thicksim Bio(\mu, N)$，此时的推导过程和 Bernoulli 其实极其相似，结果也一致。假设 $m$ 为 N 次独立 Bernoulli 试验中，$y=1$ 的次数，推导如下：

- 写出似然函数：$$L=p(m|\mu,N)=Bio(m|\mu,N)=\frac{N!}{m!(N-m)!}\mu^m(1-\mu)^{(N-m)}$$
- 似然函数取对数化简整理（阶乘项作为 constant 由 $\alpha$ 替换）：$$l=ln\ p(m|\mu,N)=\alpha+mln\mu+(N-m)ln(1-\mu)$$
- 最大化似然函数，取导数为0的极值：$$\frac{\partial\ l}{\partial\ \mu}=\frac{m}{\mu}-\frac{N-m}{1-\mu}=\frac{1}{\mu(1-\mu)}(m-\mu N)=0\\\mu_{MLE}=\displaystyle\frac{m}{N}$$

结果与 Bernoulli 完全一致，我们把 N次独立 Bernoulli trials 看作分开的一系列 Bernoulli 分布，或者一个 Binomial 分布，分析的最终结果是等价的。

### 2.3 Categorical （Multinomial N=1）
同样地，假设我们得到了对于多元（K种可能结果）随机变量 $y$ 的观察结果 $D=\{y_1,y_2,...\ ,y_N\}$，假设 $y \thicksim Mult(\mu, 1)$，这里最大的区别是要并行 $K$ 个判断，$y$ 和 $\mu$ 都是长度为 $K$ 的向量。以下标准的三步法 最大似然估计 MLE：

- 写出似然函数（设 $m_k$ 为观察到 $y_n=k$ 的数量）：$$L=p(D,\mu)=\prod_{n=1}^Np(y_n|\mu)=\prod_{n=1}^N\prod_{k=1}^K\mu_k^{x_{nk}}=\prod_{k=1}^K\mu_k^{m_k}$$
- 似然函数取对数化简整理：$$l=ln\ p(D|\mu)=\sum_{k=1}^Km_kln\mu_k$$
- 最大化似然函数，这里需要额外注意，$\mu_k$ 有一个 $\sum\mu_k=1$ 的约束，所以使用拉格朗日乘子法引入 Lagrange multiplier $\lambda$ 来包含这一约束，所以最大化的目标变为：
$$Lagrange=\sum_{k=1}^K\{m_kln\mu_k\}+\lambda(\sum_{k=1}^k\mu_k-1)\\\frac{\partial\  Lagrange}{\partial\ \mu_k}=\frac{m_k}{\mu_k}+\lambda=0,\ \mu_k=-\frac{m_k}{\lambda}\\consider\ the\ constraint,\ \sum_{k=1}^k\mu_k=-\frac{1}{\lambda}\sum_{k=1}^km_k=1, \lambda=-N\\finally,\ \mu_k^{MLE}=\frac{m_k}{N}$$

Categorical 类似 Bernoulli，也符合直观认知，另外也以之前 Bernoulli -> Binomial 的方式，推广到 Multinomial，结果是一致的，这里不再展开。

## 3. Conjugacy 共轭分布

### 3.1 Defined by Bayes
考虑贝叶斯理论：
$$posterior=\frac{likelihood*prior}{evidence}$$
上式中对于属于 指数族分布 exponential family 形式的 likelihood 函数，我们都能找到一个 共轭先验分布 conjugate prior，使得 prior 和 posterior 的属于同一种分布，相关函数形式一致。通常这个 conjugate prior 也属于 exponetial family。另外一点，这里 $evidence$ 只是一个 constant，由 likelihood 和 prior 直接决定。

Beta 是 Bernoulli、Binomial、Negative Binomial 的共轭先验分布；
Dirichelet 是 Categorical、Multinomial 的共轭分布。

### **3.2 Beta 与 Conjugacy**
该分布定义为：$$Beta(\mu|a,b)=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$

该分布描述的随机变量 $\mu$ 是一个 $[0,1]$ 的正实数，通过两个正实数的参数 $a$、$b$ 定义。另外，$\mu$ 的期望为 $E[\mu]=\displaystyle\frac{a}{a+b}$。

注意，Beta 作为 Bernoulli 共轭先验分布，可以描述 $\mu$ 本身的分布（即概率本身的概率分布），这正是 Bayesian treatment 的核心所在：不相信有确定的模型存在，模型本身也服从一个分布。
另外，这里的 $a$、$b$ 可以看做是 Binomial 的 $y=1$ 次数 和 $y=0$ 次数（分别记为 $m$、$l$），这里再看一下 Binomial 的分布：
$$Bin(m|N,\mu)=\frac{N!}{m!(N-m)!}\mu^m(1-\mu)^{(N-m)}=\frac{(m+l)!}{m!\ l!}\mu^m(1-\mu)^{l}$$
很明显和 Beta 分布的形式相似，而 Beta 分布使用的 Gamma函数 $\Gamma$ 其实可以看做是阶乘推广到实数的计算。正因为这种相似构造，所以 Beta 是 Binomial 的共轭先验，俩者，可以先看一下他们的乘积：
$$Bin(m|N,\mu)\ Beta(\mu|a,b)=p(m|N,\mu)\ p(\mu|a,b)=p(m,\mu|N,a,b)=p(m,\mu|l,a,b)=\frac{(m+l)!}{m!\ l!}\mu^m(1-\mu)^{l}\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}=c\ \mu^{a+m-1}(1-\mu)^{b+l-1}$$
上式把复杂的阶乘以及 Gamma 函数替换为一个 constant $c$，可以看出 Binomial 和 Beta 的乘积结果异常简洁，都基于 $\mu^x(1-\mu^x)$ 这个结构，我们再进一步结合 Bayesian theory 来看：
$$p(\mu|m,l,a,b)=\frac{p(m,\mu|l,a,b)\ p(\mu|a,b)}{p(m|l,a,b)}=\frac{Bin(m|N,\mu)\ Beta(\mu|a,b)}{\int_0^1\{Bin(m|N,\mu)\ Beta(\mu|a,b)\}d\mu}=\frac{\mu^{a+m-1}(1-\mu)^{b+l-1}}{constant}=Beta(\mu|a+m,b+l)$$
**以上就是 Beta 作为 Binomial 共轭先验分布的完整表述和证明！**以 Beta 为 prior，Binomial 为 likelihood，得到的 posterior 还是一个 Beta 分布，实际上仅仅是参数 $a$、$b$ 进行加法就搞定了。
以上稍显tricky的一步推导是：用于 normalization 的 constant 部分被我们直接省略了；这是因为我们发现分子 $\mu^{a+m-1}(1-\mu)^{b+l-1}$ 已经构成了 Beta 的雏形，又知道 Beta 通过其系数（三个 Gamma 函数）可以完成 normalization，所以直接推断出了最终的结果是 Beta。其实这一步也可以通过展开积分项精确证明，详见 https://stats.stackexchange.com/questions/181383/understanding-the-beta-conjugate-prior-in-bayesian-inference-about-a-frequency 。
这里涉及的概率转换的过程相当于以下任意一个公式：

- $posterior=\displaystyle\frac{likelihood*prior}{evidence}$
- $p(\mu|m)=\displaystyle\frac{p(m,\mu)\ p(\mu)}{p(m)}$
- $Beta=\displaystyle\frac{Bin\ *\ Beta}{constant}$

Dirichelet 和 Beta 类似，是把俩种结果推广到K种结果，这里不展开详述了。

### 3.3 MLE (Maximum Likelihood Estimate) to MAP (Maximum A Posterior)
这里结合实例，讲解 Beta 分布的实践应用，实际上这是从 MLE 到 MAP 的思路转换（另外一篇文章里面结合 regularization 的角度详述过俩者关联，频率派 vs. 贝叶斯派）。

例如我要估计一个硬币的抛硬币正面向上概率 $\mu$，通过本文 2.1 或者 2.2 详细描述的 Bernoulli 和 Binomial 的 MLE 求解方法，我们可以根据一系列的实验结果 $D=\{y_1,y_2,...\ ,y_N\}$（一堆1和0表示是否是向上），估计出这个 $\mu$。假设我们观察到 $D=\{1,1,1,1,1\}$，也就是我们连续5次抛硬币都是正面；这时 MLE 的估计结果为（依据 2.2 中推导）：
$$Bin(m=5|N=5,\mu),\\\mu_{MLE}=1.0$$

也就是说我们估计硬币向上的概率 100%！这个结论明显太激进，属于 overfitting，很可能是因为我们采样数据太有限造成的。为了避免 overfitting，常规的思路就是由 MLE 转为 MAP。

我们这里使用 Beta 作为 prior，我们的先验知识其实由 Beta 分布的 $a$、$b$ 参数描述。粗略地说，这俩个参数之间的关系表达出我们对 $\mu$ 的估计的倾向性，若 $a$ 大于 $b$ 就认为 $\mu$ 更倾向于1 and vice versa；并且 $a$、$b$ 取值的大小表现出我们对这个 prior 的信心，是否很容易被之后的试验结果动摇。
<center>
<img src="http://img.blog.csdn.net/20170711225825326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQ2RkMnhk/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" width="50%" height="50%" />
Figure 1. Probability Density Function of Beta Distribution
</center>

Beta 的概率密度分布函数（也就是 $\mu$ 的分布）随不同参数如 Figure 1. 所示。假设我们对抛硬币设置 $Beta(\mu|a=5,b=5)$ 作为 prior，则 MAP 的估计结果如下：
$$Beta(\mu|a=10,b=5)=\displaystyle\frac{Bin(m=5|l=0,\mu)\ *\ Beta(\mu|a=5,b=5)}{constant},\\\mu_{MAP}=\frac{10}{10+5}=0.67$$
MAP 显然比 MLE 平缓很多，因为其权衡了 prior 和 likelihood，至于到底有多敏感，可以根据参数调控。
**这里还有非常有意思的反直觉的推论：如果你连续抛一枚硬币，发现都是正面朝上时，应该如何预测一下一次结果？应该预测正面朝上。**
我们最初的先验认为正反面概率一致，但是后续的观察产出的后验表示朝上的概率更高，或许这枚硬币本身不均衡呢？所以我们下一次的预测应该基于后验来判断。



