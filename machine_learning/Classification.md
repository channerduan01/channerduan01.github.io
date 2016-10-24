#Classification
这里介绍一些经典 Classification 模型

#Logistic Regression
虽然号称 Regression，但其实是 Classification 的模型。这一模型非常经典，常用。  
##判别公式：$f(x) = sig(\theta^Tx),\ \ sig(z) = \displaystyle\frac{1}{1+e^{-z}}$  
实际上就是在经典的线性回归公式上嵌套了sigmoid（亦称为logistic）函数，把输出平滑地映射到(0,1)的区间上，产出有概率意义的分类器。常规地说，$f(x) > 0.5$ 列为 positive，反之列为 negtive；当然，这个阈值可以根据具体业务具体情况了设置。  

##梯度下降目标：$E = -ylog(f(x))-(1-y)log(1-f(x))$  
这个目标函数特别有意思，核心在于它引入log来优化 cost 计算，这一设计使得完全错误分类时的 penalty 达到无穷大，  

##学习公式：$\theta = \theta + \eta \sum{(y-f(x))x}$
这个梯度下降的学习公式，其实和 Linear Regression 的一致，他们 cost 函数的导数  $\displaystyle\frac{\partial E}{\partial \theta}$ 是一致的，很有趣

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







