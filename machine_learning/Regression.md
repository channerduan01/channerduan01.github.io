#Regression

#几个常见的损失函数
A 标识 actual value，F 标识 forecast value；
####MSE（Mean Square Error）
$$M = \displaystyle\frac{1}{n}\sum_{t=1}^n(A_t-F_t)^2$$
####MAE（Mean Absolute Error）
$$M = \displaystyle\frac{1}{n}\sum_{t=1}^n|A_t-F_t|$$
####MAPE（Mean Absolute Percentage Error）
$$M = \displaystyle\frac{1}{n}\sum_{t=1}^n\displaystyle\frac{|A_t-F_t|}{|A_t|} $$

#Linear Regression（线性回归）
常采用 MSE 的损失函数：$E = \displaystyle\frac{1}{2N}\sum^N_{i=1}(h_{\theta}(x_i)-y_i)^2$  
h 指的是 hypothesis（假设模型），就是我们用于逼近现实规律的模型，线性回归： **$h_{\theta}(x) = \theta^tx$**  
Gradient descent 梯度下降 又称为 Steepest descent，使用一阶梯度（牛顿法使用了二阶梯度信息，计算量大而更有效；相关求解方法还有很多，例如 拟牛顿法、共轭梯度法）：$\displaystyle\frac{\partial E}{\partial \theta} = (\theta^tX-y)X^t$  
相应的迭代公式：$\theta^{(t+1)} = \theta^{(t)} - \eta(\theta^tX-y)X^t$

##Least Square（最小二乘法）
直接计算线性回归的最优解，继续上面的栗子，就是：$\theta^t = X^ty(XX^t)^{-1}$；就是直接通过 $\displaystyle\frac{\partial E}{\partial \theta} = 0$，取得极值处的 $\theta$ 值  
这种方法在样本量、特征量大的真实问题中，无法完成矩阵求逆

#Ridge Regression（岭回归）
这一方法添加了 regularization item 来优化过拟合，以损失部分信息、降低精度为代价获得更 general 的回归系数。  
Cost 函数为：$E = \displaystyle\frac{1}{2N}\sum^N_{i=1}(h_{\theta}(x_i)-y_i)^2+\displaystyle\frac{1}{2}||\theta||^2$

####Regularization 的本质是优化了病态XXX

#Lasso回归
类似 Ridge Regression，对权值向量增加了 penalty，不过这个增加的是 norm-1 的绝对值形式 可以用于 Feature Selection（变量选择）  
Cost 函数为：$E = \displaystyle\frac{1}{2N}\sum^N_{i=1}(h_{\theta}(x_i)-y_i)^2+\displaystyle\frac{1}{2}|\theta|$



