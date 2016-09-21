#机器学习的本质？


#函数与反函数 (mapping & inverse mapping)
从最简单的 Linear Regression 来看，我们拥有特征空间上的矩阵 observations $X$，以及预测、回归的目标向量（或矩阵） $y$。这些 observations 是在我们确认到 $y$ 的前提下，做出的 Measurement，属于映射（mapping）。基于上述数据，我们意图构造线性模型，将 $X$ 反向映射（inverse mapping）到目标 $y$：$y^\prime=X\omega$。  


这一问题我们通常使用 凸优化（convex optimization） 的 最小二乘法（least square method）来求解模型参数 $w$。事实上，这一过程虽然常使用 梯度下降（gradient descent），但是其最优解也包含了 inverse 的过程： 
cost function: $$E = \frac{1}{2}(X\omega-y)^2\\
\nabla E$$

#概率与逆概率 （probability & inverse probability）



