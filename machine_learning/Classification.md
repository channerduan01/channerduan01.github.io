#Classification
这里介绍一些经典 Classification 模型

#Logistic Regression
虽然号称 Regression，其实常用于 Classification 。这一模型非常经典，属于 Maximum Entropy Model 在类数目为2时的特例  
##判别公式：$h(x) = sig(\theta^Tx),\ \ sig(z) = \displaystyle\frac{1}{1+e^{-z}}$  
实际上就是在经典的线性回归公式上嵌套了sigmoid（亦称为logistic）函数，把输出平滑地映射到(0,1)的区间上，产出有概率意义的分类器。常规地说，$f(x) > 0.5$ 列为 positive，反之列为 negtive；当然，这个阈值可以根据具体业务具体情况了设置。LR也常用于 CTR（Click-Through-Rate）的估计，评估广告被点击的概率  

LR这种线性模型很容易并行化，处理上亿条训练样本不是问题，但线性模型学习能力有限，需要大量特征工程预先分析出有效的特征、特征组合，从而去间接增强LR的非线性学习能力  

##数学根源：广义线性模型 \*
其实 Logistic Regression、Linear Regression 都属于 Generalized Linear Model（广义线性模型），他们的目标函数都是由概率意义推导而出的。  
对于 Logistic Regression 来说，这是一个 Bernoulli distribution（伯努利分布，0-1分布）问题，表述为：$p(y;\phi) = \phi^{y}(1-\phi)^{y-1},\ y\in{0,1}$，其中$\phi$表示样本为positive的概率。  
  
而 The exponential family distribution（指数族分布）的标准形式通过 T、a、b 确认参数为 $\eta$ 的指数族分布：
$p(y;\eta)=b(y)exp(\eta^TT(y)-a(\eta))$  
- $\eta$ 是 natural parameter  
- $T(y)$ 是充分统计量  
- $e^{-a(\eta)}$ 用作归一化；  
   
而对于 Bernoulli 来说，指数化形式为：
$p(y;\phi) = \phi^{y}(1-\phi)^{y-1}\\
 = exp\{ylog(\phi)+(1-y)log(1-\phi)\}\\
 = exp\{ylog(\displaystyle\frac{\phi}{1-\phi})+log(1-\phi)\}$  
- T(y) = y  
- $\eta = \displaystyle\frac{\phi}{1-\phi}$  
- $a(\eta) = log(1+e^\phi)$  
- b(y) = 1  
  
**最终：$\phi = \displaystyle\frac{1}{1+e^{-\eta}}$，直接导出了 sigmoid 公式！**

##梯度下降目标：$E = \displaystyle\frac{1}{N} \sum_{i}^{N}\{-y_ilog(h(x_i))-(1-y_i)log(1-h(x_i))\}$  
这个目标函数特别有意思，核心在于它引入log来优化 cost 计算，这一设计使得完全错误分类时的 penalty 达到无穷大；  
其实，这一目标函数是通过极大似然推导而出的  

##学习公式：$\theta = \theta + \alpha\displaystyle\frac{1}{N}\sum_{i}^{N}{(y_i-h(x_i))x_i}$
这个梯度下降的学习公式，其实和 Linear Regression 的一致，他们cost函数的导数  $\displaystyle\frac{\partial E}{\partial \theta}$ 一致；

#Softmax Regression
Sigmoid 函数用于 binary classification 中；而这里的 Softmax 函数用于 multiple classification，相当于前者的拓展版；另外这一个函数也广泛应用于神经网络分类问题的输出层  

从效果上说，原始的 vector 经过 Softmax 之后，将会被 normalize，所有元素不为负且和为1，具备概率意义。



#Naive Bayes


#Perceptron
感知器的分类，仅仅找到一条能切分正负样本的线（或者高维空间中的超平面），与 SVM 找到最优切分（最大 margin）不同，这个方法的容易有 overfitting，不易于 generalization，但是计算非常简单。
##判别公式：$f(x) = \theta^Tx$  
这个判别很简洁，$f(x) > 0$ 视为 positive，反之视为 negtive。
##梯度下降目标：$E = -\sum{yf(x)}$
实际上，按照判别式的设计，我们希望所有的数据满足 $yf(x) > 0$，y 是我们的 label，+1 标识 positive sample，-1 标识 negative sample。所以，很自然，简洁地推出 cost 为上述公式。

#SVM（Support Vector Machine）
这是一个非常优秀、常用的分类器，与kernel结合使用展现强大威力
##判别公式：$f(x) = \omega^t x + b$
其实判别公式和 Perceptron 的一致，只是展开了而已；然后，模型优化的目标完全不一样。向量 $\omega$ 和偏移 $b$ 可以确认出一个超平面来切割正负样本，SVM 的目标是找到的超平面和正负样本都保持最大间距。  
**核心在于，模型训练时的 constraint：$y\ (\omega^t x + b) > 1$，标签正例 $y = 1$，负例 $y = -1$**  
  
$\omega^t x + b$ 可以视为数据 $x$ 在 $\omega$ 方向上的投影长度与 $||\omega||$ 的积；所以 $x$ 到超平面的距离为：$\displaystyle\frac{|\omega^t x + b|}{||\omega||}$  
显然，在我们的 constraint 限制下，数据到超平面的最小距离为：$\displaystyle\frac{1}{||\omega||}$  
所以，我们的训练目标就变成了在 constraint 的条件下，最大化这个距离，也就是最小化 $||\omega||$
  
常用的优化目标由 Laplacian 推导推导：$L(\omega,b,\alpha) = \displaystyle\frac{1}{2}||\omega||^2-\sum_{n=1}^N{\alpha_n(y_n[\omega^t x + b] - 1)},\ \alpha_n > 0$  
最终等效于一个二项式问题：$\min_{\alpha}\displaystyle\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^Na_ia_jy_iy_jx_ix_j - \sum_{k=1}^N{\alpha_k}，\alpha_n >= 0，\sum_{n=1}^N\alpha_ny_n = 0$  
$\alpha_n > 0$ 对应的数据构成了支持向量，他们参与确定了最终超平面，所以这一模型命名为 Support Vector Machine；当数据投射到高维空间中，将会有很多数据参与确定超平面

##Kernel Trick
当我们处理复杂问题时，我们希望把原始的数据非线性地投射到高维度空间，甚至是无穷维度的空间来寻找完美分割数据的超平面，我们使用投影函数 $\phi(x)$ 改写优化目标中 $x$ 的内积为：$\phi^T(x_i)\phi(x_j)$。但是，这种操作会引入大量的计算，例如投射到无穷维度根本无法完成。这里使用 Kernel 将问题等效为：$K(x_i,x_j)$，**我们不需要投影到高维空间再计算内积，而是直接计算当前特征空间中的内积，再通过 Kernel 函数求出其在高维空间中对应的内积**。对于不同问题，不同场景，可以尝试选用不同的 Kernel 函数。Kernel函数 也是当下研究的一大热点。

##SVM 和 LR 的差异
- loss函数不同；LR（Logistic Regression）采用 log损失，SVM 采用 hinge（合页）损失。
- LR 考虑全局，寻找一个所有点都远离的超平面；而 SVM 只考虑让 support vectors（支持向量） 远离而最大化 margin；所以 LR 对异常值敏感，而 SVM 对异常值不敏感。
- 在训练集较小时，SVM 较适用，而 LR 需要较多的样本；SVM 推广能力更强。
- 对非线性问题的处理方式不同；LR 主要靠特征构造，必须组合交叉特征，特征离散化。SVM也可以这样，还可以通过 kernel。





