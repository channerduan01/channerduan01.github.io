#Regression

#几个常见的损失函数
A 标识 actual value，F 标识 forecast value；
##MSE（Mean Square Error）
$$M = \displaystyle\frac{1}{n}\sum_{t=1}^n(A_t-F_t)^2$$
##MAE（Mean Absolute Error）
$$M = \displaystyle\frac{1}{n}\sum_{t=1}^n|A_t-F_t|$$
##MAPE（Mean Absolute Percentage Error）
$$M = \displaystyle\frac{1}{n}\sum_{t=1}^n\displaystyle\frac{|A_t-F_t|}{|A_t|} $$

#Linear Regression（线性回归）
$E = \displaystyle\frac{1}{2N}\sum^N_{i=1}(h_{\theta}(x_i)-y_i)^2$  
h 指的是 hypothesis（假设模型），就是我们用于逼近现实规律的模型

##Least Square（最小二乘法）

#Ridge Regression（岭回归）
这一方法添加了 regularization item 来优化过拟合，以损失部分信息、降低精度为代价获得更 general 的回归系数

##regularization 的本质是优化了病态XXX




#LASSO回归
可以用于 Feature Selection（变量选择）

