# Think of LOSS



## Gradient Descent

- The first step is to define a loss that is going to be minimized.
- Then, calculated the gradient of loss function. The gradient is the direction that makes loss bigger.
- Finally, update the parameters of model by adding the product of learning rate and the negative gradient. For right learning rate, the loss descends.


## Cross Entropy
交叉熵衡量了两个概率分布的差异（模型预测分布 vs. 样本真实分布）。交叉熵越小，模型预测效果越好。
在衡量模型的场景下，交叉熵等价于相对熵（即KL散度）。

https://blog.csdn.net/weixin_40485472/article/details/81131446


tensorflow 框架中习惯使用函数 sparse_softmax_cross_entropy_with_logits 来直接产出 loss，这个函数的输入是模型预测的原始 logits，其函数内部计算了 softmax 并应该做了针对性优化（只计算真实分布 label 标明的分类的概率），然后计算 cross entropy 作 loss。

[这是一个很好的举例说明](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)



