#机器学习的本质
这里从俩个抽象视角看到机器学习问题

##函数与反函数（mapping & inverse mapping）
从最简单的 Linear Regression 来看，我们拥有特征空间上的矩阵 observations $X$，以及预测、回归的目标向量（或矩阵） $y$。这些 observations 是在我们确认到 $y$ 的前提下，做出的 Measurement，属于映射（mapping）。基于上述数据，我们意图构造线性模型，将 $X$ 反向映射（inverse mapping）到目标 $y$：$y^\prime=X\omega$。  

这一问题我们通常使用 凸优化（convex optimization） 的 最小二乘法（least square method）来求解模型参数 $w$。事实上，这一过程虽然常使用 梯度下降（gradient descent），但是其最优解也包含了 inverse 的过程： 
cost function: $$E = \frac{1}{2N}(\theta^tX-y)^2$$
我们尝试猜测出一个合适的反函数，去逼近真正的自然规律、上帝函数

##概率与逆概率（probability & inverse probability）
类似于上述提及的 映射 与 反映射，这也能被看做一个 概率 与 反概率 的过程。
$$P(y_j|X) = \displaystyle\frac{P(X|y_j)P(y_j)}{\sum_i^KP(X|y_i)P(y_i)}$$
我们基于确认到 $y$ 之后，测量 P(X|y) 的分布，来反向推断不同特征 $X$ 所造成的结果 P(y|X) 

#奥卡姆剃刀定律（Occam's Razor）
**如无必要，勿增实体（Entities must not be multiplied beyond what is necessary）**，尽量用最简单的模型；因为 underfitting 和 overfitting 是一个悖论  
越复杂，描述能力越强的模型，越容易 overfitting，Good performance on training data does not (necessarily) mean good performance on test data  
这一定义广泛应用到了机器学习各个模型的训练中，例如 正则项（regularization）会给予weight vector惩罚，使得训练目标由最小化经验风险转为最小化结构风险，权衡出较简单的模型来解决问题。Tree模型中也有大量的剪枝策略，来简化模型，提高模型推广、泛化能力。


#“没有免费的午餐”（No Free Lunch）定理


