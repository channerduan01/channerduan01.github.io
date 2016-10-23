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
##


##Kernel Trick








