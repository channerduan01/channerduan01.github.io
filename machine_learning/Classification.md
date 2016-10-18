#Classification
这里介绍一些经典 Classification 模型

#Logistic Regression
虽然号称 Regression，但其实是 Classification 的模型。这一模型非常经典，常用。  
##判别公式：$f(x) = sig(\theta^Tx),\ \ sig(z) = \displaystyle\frac{1}{1+e^{-z}}$  
实际上就是在经典的线性回归公式上嵌套了sigmoid（logistic）函数，把输出平滑地整合到(0,1)的区间上，产出有概率意义的分类器。常规地说，f(x) > 0.5 列为 positive，反之列为 negtive；当然，这个阈值可以根据具体业务具体情况了设置。  
##梯度下降目标：$E = y$  
这个目标函数特别有意思，核心在于它引入log来优化 cost 计算，这一操使得完全错误分类时的 penalty 达到无穷大。


#Perceptron



