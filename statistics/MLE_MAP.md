# MLE 和 MAP  
Abstract: 常用的基于统计的机器学习基本方法，很多经典的机器学习算法都能基于他们找到解释。MLE属于频率学派，MAP属于概率学派

## MLE (Maximum Likelihood Estimation) 最大似然估计
给定一组采样（数据），他们都是从同一个分布中采样（每次采样的独立的）出来的。我们不知道其具体的分布，但是我们推测它属于某个分布族，我们只是需要确定具体参数即可，即”模型已定，参数未知”。这时，最大似然估计 就可以用来估计模型参数，就是找出一组参数，使模型产出观测数据的概率最大。例如，我们确定分布式高斯分布的话，我们只需要确认其均值和方差就可以了。

构造似然函数（likelihood）如下，就是在某一模型参数$\theta$条件下，模型产出观测数据的条件概率：
$$L(\theta;x_1,x_2,...,x_n)=f(x_1,x_2,...,x_n|\theta)=\prod_{i=1}^n{f(x_i|\theta)}$$
因为每个采样都是独立的，所以上式中我们可以把联合概率的概率密度函数$f(x_1,x_2,...,x_n|\theta)$直接分开为单个采样概率密度函数的乘积。另外，实际中我们常用log-似然函数（log-likelihood），以便于做一些代数处理。最终的目标形式如下，找到使模型产出观测数据的概率最大的模型参数 $\hat{\theta}_{MLE}$：

$$\hat{\theta}_{MLE}=arg\max_{\theta}{f(x_1,x_2,...,x_n|\theta)}=arg\max_{\theta}{ln(L)}$$

## MLE and Least Square
对于基本的 Linear Regression 模型：$x^Tw=y$，定义模型的误差为$\epsilon=y-x^Tw$，然后我们假设误差服从如下高斯分布$\epsilon \thicksim N(0,\sigma^2)$，则有我们的观测服从高斯分布$y\thicksim N(x^Tw, \sigma^2)$
高斯分布的概率密度函数（probability density function）如下：
$$f(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi^2}} \exp(-\frac{(x-\mu)^2}{2\sigma^2})$$
可定义我们的模型上，观察到结果y的概率密度函数如下：
$$p(y|x,w,\sigma)=\frac{1}{\sqrt{2\pi\sigma}}\exp(-\frac{(y-x^Tw)^2}{2\sigma^2})$$
由MLE，定义似然函数如下：
$$\ln L(w,X,\sigma)=ln\ \prod_{i=1}^n{p(y_i|x_i,w,\sigma)}=\sum_{i=1}^n{ln\frac{1}{\sqrt{2\pi\sigma}}\exp(-\frac{(y_i-x_i^Tw)^2}{2\sigma^2})}$$
$$=nln\frac{1}{\sqrt{2\pi\sigma}}-\frac{1}{2\sigma^2}\sum_{i=1}^n{(y_i-x_i^Tw)^2}$$
上式结果中，考虑取一个 $w$ 使得似然函数最大： 
$$\hat{w}_{MLE}=arg\min_{w}{\sum_{i=1}^n{(y_i-x_i^Tw)^2}}$$
这个解和Least Square（最小二乘法）完全一致

## MAP (Maximum a Posterior Estimation) 最大后验估计
MAP给MLE增加了一个关于参数本身的priordistribution（先验分布），仅仅比MLE多了这一步。MAP是求使后验概率最大的模型参数，其利用Bayes公式，最大化如下的后验概率，即给定了观测数据后使参数概率最大：
$$P(\theta|x_1,x_2,...,x_n)=\frac{P(x_1,x_2,...,x_n|\theta)P(\theta)}{P(x_1,x_2,...,x_n)}$$
**当先验分布是uniform distribution时，俩者是一致的。**然而，他们一个属于频率学派（MLE），一个属于贝叶斯学派（MAP）~MAP的最优解如下： 
$$\hat{\theta}_{MAP}=arg\max_{\theta}{f(\theta|x_1,x_2,...,x_n)}=arg\max_{\theta}{f(x_1,x_2,...,x_n|\theta)g(\theta)}$$

##MAP and Ridge Regression
我们在用MLE得到Least Square的基础上，对MLE加入高斯先验分布，假设我们要求解的参数$w\thicksim N(0,\gamma^2)$，这时，我们最大化的目标函数如下：
$$L(w)=p(y|X,w)p(w)=\prod_{i=1}^n{\{\frac{1}{\sqrt{2\pi\sigma}}\exp(-\frac{(y_i-x_i^Tw)^2}{2\sigma^2})\}}\prod_{j=1}^m{\{\frac{1}{\sqrt{2\pi\gamma}}\exp(-\frac{(w_j)^2}{2\gamma^2})\}}$$
log后化简后结果如下：
$$lnL(w)=nln\frac{1}{\sqrt{2\pi\sigma}}+mln\frac{1}{\sqrt{2\pi\gamma}}-\frac{1}{2\sigma^2}\sum_{i=1}^n{(y_i-x_i^Tw)^2}-\frac{1}{2\gamma^2}w^Tw$$
可以得出:
$$\hat{w}_{MAP_{Gassian}}=arg\min_{w}{\sum_{i=1}^n{(y_i-x_i^Tw)^2}}+\lambda||w||^2$$

##MAP and LASSO（least absolute shrinkage and selection operator）
对MLE加入拉普拉斯先验分布，拉普拉斯分布（Laplace Distribution）概率密度公式如下：
$$f(x|\mu,b)=\frac{1}{2b} \exp(-\frac{|x-\mu|}{b})$$

类似之前Ridge的变化过程，可以得到最后的solution是：
$$\hat{w}_{MAP_{Laplace}}=arg\min_{w}{\sum_{i=1}^n{(y_i-x_i^Tw)^2}}+\lambda||w||^1$$
和LASSO一致
