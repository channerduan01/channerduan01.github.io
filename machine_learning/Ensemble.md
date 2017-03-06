#Ensemble
指的的集合一组训练模型，来共同决策的方式；显然，组合模型越多将会显得更加耗时，也会带来更多 generalization 的能力

#Random Forest
广泛使用的随机森林，是以一系列的 Decision Tree 构成的。 Decision Tree 最有趣的地方是它和神经网络一样是层次结构（当然它简单的多），它通过寻找信息增益最大化的特征来构建模型，而神经网络寻找残差最小的方式构建模型

###Decision Tree
以下列举相关重要算法，决策树是非线性的模型
####ID3，分类树
训练基于 Entropy 熵：
$$E(x) = -\sum_{i=1}^np_ilog_2p_i$$
这里的 $p_i$ 表示某一个成分所占比例；熵表示的是系统的不确定程度，越大越不确定
Information Gain：
$$Gain = E(U)-(P(u_1)E(u_1)+P(u_2)E(u_2))$$
这里的 E 表示 Entropy；  
U 表示分割前的集合；  
$u_1、u_2$ 表示分割后的集合；  
$P(u_1)、P(u_1)$ 表示分割后子集占的比例，他们和为 1
####C4.5
是对于 ID3 的优化，可以处理连续型数据（处理方法其实也是尝试分片，搜索尝试阈值）
####Classification And Regression Tree（CART）
- 对于 分类树 基于 基尼纯净度（Gini impurity）进行切分
- 对于 回归树 基于 方差缩减（variance reduction）进行切分；对于回归树来说，每一个节点都用均值来代表一个值


#GBDT（Gradient Boost Decision Tree）
**核心点：这里面涉及的树都是 Regression Decision Tree 回归树，而不是 分类树！回归树的结果叠加是有意义的，而分类树的结果叠加没有意义；GBDT 的结果是所有回归树结果的叠加。**  
GBDT 的核心过程在于，后续的树，基于之前树预测的残差进行学习。比如某个人 10岁，第一个数的训练结果是 14岁，那么第二棵树的训练目标就是 -4岁；这样子我们最终叠加所有树的预测结果，就能得到相对robust的结果。  
这里面还是体现Boosting的核心思想：先解决主要问题，再根据反馈逐渐优化结果  

CTR（Click-Through-Rate）预估时，常采用 GBDT+LR 或者 GBDT+FM 的方式；这种场景下，GBDT 完成了 feature selection，每个原始样本将被 GBDT 激活情况的 01序列 所重新代表（每颗回归树仅激活一个1）；所以新的样本空间维度为 每棵树叶子树*树数量

#Adaptive Boost
典型的提升方法。我之前的人脸检测项目在 Cascade 结构中使用过

