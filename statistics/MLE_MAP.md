#MLE (最大似然) 与 LS (最小二乘) 与 MAP (最大后验)

##序言
最大似然估计 属于机器学习中的常用的基础思想，很多具体的算法及模型都基于它建立，或者能够基于它找到解释，例如：

- MLE 可以解释为什么常用的线性回归使用的是平方（即最小二乘法），而不是四次方

- MLE 思想 与 MAP 思想的联系与区别；这关于概率统计领域 频率学派 vs. 贝叶斯学派；还会涉及到对于机器学习中 Regularization 的理解；（MAP 与 贝叶斯估计，朴素贝叶斯分类器 乃至 Logistic Regression LR 都相关，这些内容其他文章再展开讨论）

- MLE 思想，被应用于机器学习十大算法之一 EM算法（期望最大化，K-means 实际上使用了 EM；EM 其他文章再展开讨论）


本文将会详细阐述 最大似然 的思想，并展开讨论 LS 、MAP 与最大似然的关联。


## 1. MLE 最大似然估计 
MLE (Maximum Likelihood Estimation) 
### 1.1 问题定义（适用场景）
- 给定一组采样（数据），他们都是从**同一个分布（identically）**中采样，并且**每次采样的独立的（即独立事件，independently）**
- 我们不知道其具体的分布，但是**我们认为(推测)它属于某个分布族**，所以只需要确定具体参数即可，即 **"模型已定，参数未知"**

这时，最大似然估计 就可以用来估计模型参数，就是找出一组参数，使模型产出观测数据的概率最大。例如，我们确定分布式高斯分布的话，我们的目标只是确认其均值和方差。
*（上述定义中加粗的三个部分强调的 最大似然估计 非常强的三点假设）*
### 1.2 似然函数
定义问题后，我们使用 似然函数（likelihood） 来定量地表示**模型产出观测数据的概率**，可以理解为定量标识条件概率 $p(X|\theta)$，其中 $\theta$ 是我们想估计的模型参数，而 $X$ 是已经观测到的数据。似然函数准确定义如下：
$$L(\theta;x_1,x_2,...,x_n)=f(x_1,x_2,...,x_n|\theta)=\prod_{i=1}^n{f(x_i|\theta)}$$

- 我们通过模型的概率密度函数 $f$ 来表达 likelihood；例如 高斯分布的概率密度函数是 $f(x|\theta)=\displaystyle\frac{1}{\sqrt{2\pi^2}} \exp(-\frac{(x-\mu)^2}{2\sigma^2})$
- 由于我们假设采样是独立的，所以我们可以把基于所有采样的联合概率，拆分为 $n$ 个独立概率的积
- 实践中，常采用对数似然函数，这样在一些化简上更加方便，且最大化时是等价的；称为log-likelihood：$ln(L)=\sum_{i=1}^nf(x_i|\theta)$
### 1.3 最大似然估计
在定义问题且确定目标函数（似然函数）后，我们要做的就是最大化目标函数；也就是找到**使模型产出观测数据的概率最大的一组模型参数**  $\hat{\theta}_{MLE}$：
$$\hat{\theta}_{MLE}=arg\max_{\theta}{f(x_1,x_2,...,x_n|\theta)}=arg\max_{\theta}{ln(L)}$$

以上的 1.1 到 1.3 对最大似然估计的思想做了整理，实际中会有很多繁琐的细节来找到最优的 $\hat{\theta}_{MLE}$：

- 写出似然函数具体式子（考虑模型的概率密度公式）
- 对似然函数取对数化简整理
- 最大化似然函数（例如解使导数等于0的参数值）  

这里引入一个非常简单的具体例子：  

- 假设我们从一个简单的高斯分布$N(\mu,\sigma)$采样到了 $n$ 个样本，且满足 IID（Independently and Identically Distributed），可以写出似然函数如下：
$$L(X,\mu,\sigma)=\prod_{i=1}^n{\frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{(x_i-\mu)^2}{2\sigma^2})}$$
- 对似然函数取对数化简后如下：
$$\begin{aligned}l = ln\ L\ 
&= \sum^n_{i=1}ln(2\pi\sigma^2)^{-\frac{1}{2}}+\sum^n_{i=1}(-\frac{(x_i-\mu)^2}{2\sigma^2})\\ 
&=-\frac{N}{2}ln\sigma^2-\frac{N}{2}ln(2\pi)-\frac{1}{2\sigma^2}\sum^n_{i=1}(x_i-\mu)^2\end{aligned}$$

- 最大化上式；即分别对 $\mu$ 和 $\sigma$ 求解偏导等于 0
$$\frac{\partial\ l}{\partial\ \mu}
=\frac{1}{\sigma^2}\sum^n_{i=1}(x_i-\mu)=0\\
\sum^n_{i=1}x_i=\sum^n_{i=1}\mu\\
\hat{\mu}=\frac{1}{N}\sum^n_{i=1}x_i$$
$$\frac{\partial\ l}{\partial\ \sigma^2}
=-\frac{N}{2\sigma^2}+\frac{1}{2(\sigma^2)^2}\sum^n_{i=1}(x_i-\mu)^2=0\\
\frac{1}{2\sigma^2}(\frac{1}{\sigma^2}\sum^n_{i=1}(x_i-\mu)^2-N)=0\\
\hat{\sigma^2}=\frac{1}{N}\sum^n_{i=1}(x_i-\hat{\mu})^2$$

以上三步通过 最大似然估计 完成了对高斯分布模型参数的估计，估计结果和均值及方差的公式完全一致。

## 2. 最大似然估计 与 LS 最小二乘法 
MLE & Least Squares
### 2.1 最小二乘法
最小二乘法 是非常标准的回归 regression 问题的解法，这里简单列举几个点：

- 通过最小化残差（residual，观测值 和 模型预测值 之差）的平方和，以寻找数据的最佳函数匹配：$\min{\epsilon=\sum_{i=1}^n{(y_i-f(x_i,\theta))^2}}$
- 其解决了 超定问题 over-constrained：样本数 大于 特征数，实际上是 方程数量 大于 未知参数数量
- 很多情况下，**最小二乘法 是 残差 满足正态分布的 最大似然估计**；这与 高斯分布（Gaussian Distribution）、中心极限定理（Central-limit Theorem）关系密切！这也直接回答了我们为什么要用最小平方而不是四次方。下面将从线性模型开始，通过 最大似然估计 推导出 最小二乘法

### 2.2 线性模型 与 残差
#### 2.2.1 定义
对于线性回归（Linear Regression）基本的模型 $g(x)=x^Tw$：

- 这里认为 $x$ 和 $w$ 分别是 观测数据（data） 和 模型参数（weight vector），$g(x)$ 是模型预测值
- 另外引入标签（label）$y$ 作为想要预测的值；注意 $x$ 和 $y$ 都是我们的观测（采样）数据，准确说他们应该是 pair 形式的观测结果 $(x, y)$
- 模型预测结果 和 标签 的差值定义为 **残差（residual）**： $\epsilon=y-g(x)$；注意，term 残差 专指模型预测差值，和 误差（又分为 系统误差 和 随机误差） 不同
#### 2.2.2 残差分布
如果我们每一次的观测都属于独立事件，所有观测误差的期望和方差应该都一致；显然这符合中心极限定理，应该构成正态分布，并且误差的期望值应该是 0。所以大多数情况下，我们可以认为这个误差服从高斯分布，如下：
$$\epsilon \thicksim N(0,\sigma^2)$$
#### 2.2.3 与 最大似然
结合本小节上述两个公式，可以得到我们的观测到的标签服从如下高斯分布：
$$y\thicksim N(g(x), \sigma^2)$$
此时，我们定义了产出观测数据的模型，处于**"模型已定，参数未知"**的情况，**找到一组参数使我们观测到一系列 $y$ 的概率最大** 是一种完成 线性回归 的很自然的思路，这也就是 **最大似然估计** !

### 2.3 最大似然估计 推得 最小二乘法
#### 2.3.1 似然函数
有高斯分布的概率密度函数（probability density function）如下：
$$f(x|\ \mu,\sigma^2)
=\frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{(x-\mu)^2}{2\sigma^2})$$
则 2.2 中观察到结果 y 的概率密度函数如下：
$$p(y|\ x,w,\sigma)
=\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(y-g(x))^2}{2\sigma^2})$$
可定义我们观测的到一系列 $y$ 的可能性（似然函数）如下：
$$L(w,X,\sigma)=\prod_{i=1}^n{p(y_i|x_i,w,\sigma)}$$
#### 2.3.2 化简整理
对似然函数取对数、展开、化简如下：
$$\begin{aligned}ln\ L(w,X,\sigma)
&=ln\ \prod_{i=1}^n{p(y_i|x_i,w,\sigma)}\\
&=\sum_{i=1}^n{ln\ p(y_i|x_i,w,\sigma)}\\
&=\sum_{i=1}^n{ln\{\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(y_i-x_i^Tw)^2}{2\sigma^2})\}}\\
&=n\ ln\{\frac{1}{\sqrt{2\pi\sigma^2}}\}-\frac{1}{2\sigma^2}\sum_{i=1}^n{(y_i-x_i^Tw)^2}\end{aligned}$$
#### 2.3.3 最大似然
上述结果中，$\sigma$ 可以假设为任意大于 0 的常数（表示残差高斯分布的标准差），对于剩下的参数部分最大化则有（因为负号所以变成最小化）：
$$\hat{w}_{MLE}=arg\min_{w}{\sum_{i=1}^n{(y_i-x_i^Tw)^2}}$$
这个解和 Least Square（最小二乘法）完全一致，即对于 Linear Regression，**最小二乘法 是 残差 满足正态分布的 最大似然估计。**在这之后，可以直接对上式求导等于 0 解出 $\hat{w}_{MLE}=X^+y$ ，或者更常规地使用 gradient descend 方法

## 3. 最大似然估计 MLE 与 最大后验估计 MAP
MLE & MAP (Maximum a Posterior Estimation) 

### 3.1 唯一差别 prior
MAP 相对 MLE 的**唯一差别**是：增加了一个关于参数本身的**prior distribution（先验分布）**。其实就是通过先验知识，把参数的解约束到一定范围内，所以和 Regularization 有很大联系。相关 贝叶斯法则如下 Bayes Rule：
$$posterior=\frac{likelihood*prior}{evidence}\ or\ P(Y|X)=\frac{P(X|Y)P(Y)}{P(X)}$$
最大后验估计 MAP 利用了 贝叶斯法则，定义最大化目标为如下的后验概率：
$$P(\theta|x_1,x_2,...,x_n)=\frac{P(x_1,x_2,...,x_n|\theta)P(\theta)}{P(x_1,x_2,...,x_n)}$$
上式中，分母 $P(x_1,x_2,...,x_n)$ 作为数据本身存在的概率（evidence，也称边界似然，即对所有可能存在的似然概率积分），可以看做是一个常数，我们的目标是最大化（通过 $\theta$）以下俩个目标 ：  

- 最大似然 MLE 的 似然函数 (likelihood): $P(x_1,x_2,...,x_n|\theta)$ 
- 参数本身的先验分布: $P(\theta)$  

自然地，**当先验分布是 uniform distribution 时，MLE 和 MAP 一致。**但其实，一个属于频率学派（MLE），另一个属于贝叶斯学派（MAP）。
继续使用 $f(x_1,x_2,...,x_n|\theta)$ 表示相关参数产出观测结果的likelihood，并引入 $g(\theta)$ 表示参数先验分布的概率密度函数，则 MAP 解如下： 
$$\begin{aligned}\hat{\theta}_{MAP}=&\ arg\max_{\theta}{f(\theta|x_1,x_2,...,x_n)}\\=&\ arg\max_{\theta}{f(x_1,x_2,...,x_n|\theta)g(\theta)}\end{aligned}$$

**一句话归纳即**：

- **MLE 寻找使模型产出观测数据的概率最大的一组模型参数**
- **MAP 寻找对于已知先验概率以及观测数据最适合的一组模型参数**

### 3.2 MAP 与 Regularization
先验分布 本质上是我们对相关领域、数据、模型的已有经验或知识。由于机器学习本身不是万能的，这种 先验知识 往往能起很大作用。实践中常提到 Regularization 正则化 来优化模型的过拟合，达到更好的泛化 generalization 表现；这本身就是通过 先验知识 对模型的求解进行约束，以下详细列举两个例子。

#### 3.2.1 MAP and 岭回归 Ridge Regression
在之前 MLE 得到 Least Square 的基础上，对MLE加入高斯先验分布，假设我们**要求解的参数 $w$ 本身服从一个先验分布：$w\thicksim N(0,\gamma^2)$**  
此时，MAP 最大化的目标函数如下：
$$\begin{aligned}L(w)&=p(y|X,w)p(w)\\&=\prod_{i=1}^n{\{\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(y_i-x_i^Tw)^2}{2\sigma^2})\}}\prod_{j=1}^m{\{\frac{1}{\sqrt{2\pi\gamma^2}}\exp(-\frac{(w_j)^2}{2\gamma^2})\}}\end{aligned}$$
注意，这里假设了观测数据 $X$ 包含 $n$ 个样本，每个样本 $m$ 个特征；所以对于一个固定的 $w$，我们的目标函数是：整个数据集上观测到残差的联合概率 与 $w$ 本身存在的概率 的积。

取对数后，化简整理得如下结果：
$$lnL(w)=nln\frac{1}{\sqrt{2\pi\sigma^2}}+mln\frac{1}{\sqrt{2\pi\gamma^2}}-\frac{1}{2\sigma^2}\sum_{i=1}^n{(y_i-x_i^Tw)^2}-\frac{1}{2\gamma^2}w^Tw$$
上式中，$\sigma$ 和 $\gamma$ 看作是某一常数，但是和 MLE 不同的是，他们的取值会影响两个目标（likelihood & prior）的权重，所以引入 超参数 $\lambda$ 来直接表示先验的权重，最终有:
$$\hat{w}_{MAP_{Gassian}}=arg\min_{w}{\sum_{i=1}^n{(y_i-x_i^Tw)^2}}+\lambda||w||^2$$
上式就是标准的 岭回归 Ridge Regression 公式（LS + L2正则）。本质上就是在 最小二乘法 的基础上增加的参数本身的先验分布，认为参数本身服从高斯分布。  
从 L2-Regularization 的角度来看，就是要求 weight vector 中大部分的权值聚集在 0 附近，正如下图中展示的 Gaussian 分布；这可以控制模型的复杂度，优化过拟合。
<center>
<img src="http://img.blog.csdn.net/20170512083857606?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQ2RkMnhk/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" width="50%" height="50%" />
Figure 1. Gaussian distribution with different parameters
</center>
#### 3.2.2 MAP and 套索回归 LASSO（least absolute shrinkage and selection operator）
和 3.2.1 类似，我们对 MLE 加入拉普拉斯分布（Laplace Distribution）为先验分布将得到 LASSO。拉普拉斯分布的概率密度公式如下：
$$f(x|\mu,b)=\frac{1}{2b} \exp(-\frac{|x-\mu|}{b})$$

与之前的似然函数展开、整理、化简过程类似，我们最终得到下式：
$$\hat{w}_{MAP_{Laplace}}=arg\min_{w}{\sum_{i=1}^n{(y_i-x_i^Tw)^2}}+\lambda||w||^1$$

从下图的 Laplace 中可以看出，这个先验分布会更加地把解约束到 0 附近，甚至就是 0，所以 L1-Regularization 也常用作稀疏（sparse）解、特征筛选等。特别是针对 拥有大规模的特征的简单模型，利用这种稀疏性的先验约束，去除冗余的噪声的特征是很重要的。
<center>
<img src="http://img.blog.csdn.net/20170512094213403?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQ2RkMnhk/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" width="50%" height="50%" />
Figure 2. Laplace distribution with different parameters
</center>
另外，L1 L2 作为 Regularization 常有下图所示的解释，来说明 L1 正则化 能构造出稀疏性解；因为俩个不同优化目标的等高线（contour）的交点会有差异，L1 的交点倾向于落在坐标轴上，直接干掉了一些维度的特征，L2 的话可能会再很多维度上一直保持一个较小的量。整个图很直观，当然相关结论我们都可以用 MAP 的推导结果直接解释，L1 本身就通过 Laplace 把参数的解约束到稀疏分布上了，L2 本身通过 Gaussian 把参数的解约束到较平滑的分布上。
<center>
<img src="http://img.blog.csdn.net/20170512094311607?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQ2RkMnhk/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" width="50%" height="50%" />
Figure 3. Regularisation effect for weight vector
</center>
### 3.3 MAP 与 贝叶斯分类器 NB (Naive Bayes Classifiers)
之前也提到，MAP 的核心在于利用 贝叶斯法则 Bayesian Rules 引入先验；这其实就与 Bayes Estimate、Bayes Classifier、Naive Bayes Classifier 有着密切联系了。甚至 LR (Logistic Regression) 以及 生成模型 vs. 判别模型 的问题都可以展开讨论（其他文章展开讨论）

## 4. 最大似然估计 MLE 与 最大期望算法 EM
MLE & EM (Expectation Maximization) 

EM 被称为机器学习十大算法之一。当模型中包含我们观察不到的隐含变量（latent varaible），我们常用 EM 求解问题。这是一个迭代算法（iterative method）；分为 E (Ecpectation) 步骤用于估计模型中的隐含变量 和 M (Maximization) 步骤用于估计模型本身的参数 (具体使用 MLE 或 MAP)。
最常见的例子就是 GMM (Gaussian Mixture Model) 中我们通过 E step 来估计样本具体归属哪个高斯分布，而 M step 来估计各个高斯分布最合适的参数。K-means 就是约束了：a. 任何样本100%归属仅一个高斯分布； b. 所有高斯分布方差一致，且都是 isotropic 的（所以距离最近的一个分布概率最大）；两个条件的 GMM 模型 EM 求解算法
这一算法可以说和上述的 MLE 和 MAP 思想都有关联，后续会专门发文详细讨论 EM。
