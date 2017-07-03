# Ensemble 融合模型
指的的集合一组训练模型，来共同决策的方式，这是一类非常经典，在实践中广泛使用的机器学习模型。其具体参与组合的模型有很多选择（常见的是决策树，当然也可以是其他各种模型），组合的方式也有很多选择：Bagging、Boosting 等等。当下非常流行的模型有：AdaBoost、GBDT、XGBoost 等等，本文将从理论上梳理一下 Ensemble 的主要脉络。

## Decision Tree 决策树

### 距离度量的概念
- 样本之间的相似性度量，KNN
- 最大间隔的线性分类器，SVM支持向量机
- 相似输入向量输出相似，LR

当然数据中拥有一些无法度量的概念，基于度量的方法难以直接使用时，可以考虑使用决策树。决策树基于一些列的节点规则来决定走到那个叶子节点，然后根据叶子节点的输出规则给出输出。



ID3

C4.5



# CART（Classification And Regression Tree）
一般采用二叉树形式
定义不纯度的下降作为分支生长方法
通常可以采用分支停止技术和剪枝技术来控制树的复杂度
实际上是一种基于规则的样本记忆方法。通过剪枝、分支停止、属性选择方式、叶子节点样本归属来增强其推广能力



# 集成思想
Ensemble 方法的可以按照以下主要思想分类：
## Bagging (Committee)
train L different models and then make predictions using the average of the predictions made by each model

### 一般来说，Bagging 更适合集成弱分类器
- 弱分类器：**不稳定**，随机采样会得到较不同的基分类器；这些基分类器准确率略高于50%。例如决策树
- 强分类器：**稳定**，随机采样对结果影响不大；反而可能不如不集成（基分类器可以由更多训练样本）。例如KNN



## Boosting
training multiple models in sequence in which the error function used to train a particular model depends on the performance of the previous models


### Gradient Boosting vs. Adaboost
模型假设：均为加性模型
优化方式：前者模型空间梯度下降方式进行优化，可以选用多种目标函数；后者指数目标函数可以通过训练分类器最小化加权错误率来直接进行优化



## Space Split
select one of the models to make the prediction, in which the choice of model is a function of the input variables.
本质上是通过对特征空间的划分，为不同input指定其使用的应用的模型，具体来说又可以分为两类切分方法：
- hard splits, Tree-based Model：将input通过一个决策树的结构，叶子节点对应其使用的模型；这相当于使用硬规则来对特征空间进行了划分，将数据分配到最合适的模型来处理。
- soft splits, Mixture of Models：一条数据归属哪个模型是一个分布，所以是一个软划分，一个input会使用其相关的一组模型。其实聚类里面也是这个思想。
例如有 Mixture of Experts (MOE)

## Boosting 和 Bagging 对比

- Adaboosting 的训练集选取与前面各轮的学习结果相关，而 Bagging 训练集选取是随机的，各轮训练集之间相互独立。且 Adaboosting 的每个分量分类器有权重，而 Bagging 的没有。


- Bagging 在大规模并行训练上有明显优势，各个模型可以并行生成；而 Boosting 方法所有模型必须按顺序生成，所以只能在并发单个模型内部的训练逻辑

# Random Forest
从 Ensemble 角度看，其使用 CART 作为子模型，通过 Bagging 集成，直接投票决定整体的最终输出。这一模型非常常见，常用作各种问题的 benchmark。


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
GBDT 的核心过程在于，后续的树，基于之前树预测的残差进行学习。比如某个人 10岁，第一棵树的训练结果是 14岁，那么第二棵树的训练目标就是 -4岁；这样子我们最终叠加所有树的预测结果，就能得到相对robust的结果。  
这里面还是体现Boosting的核心思想：先解决主要问题，再根据反馈逐渐优化结果  

CTR（Click-Through-Rate）预估时，常采用 GBDT+LR 或者 GBDT+FM 的方式；这种场景下，GBDT 完成了 feature selection，每个原始样本将被 GBDT 激活情况的 01序列 所重新代表（每颗回归树仅激活一个1）；所以新的样本空间维度为 每棵树叶子数量*树数量

#Adaptive Boost
典型的提升方法。我之前的人脸检测项目在 Cascade 结构中使用过

