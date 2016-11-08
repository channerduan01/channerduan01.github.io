#Ensemble
指的的集合一组训练模型，来共同决策的方式；显然，组合模型越多将会显得更加耗时，也会带来更多 generalization 的能力

#Random Forest
广泛使用的随机森林，是以一系列的 Decision Tree 构成的。 Decision Tree 最有趣的地方是它和神经网络一样是层次结构（当然它简单的多），它通过寻找信息增益最大化的特征来构建模型，而神经网络寻找残差最小的方式构建模型

##Decision Tree
也可以称为 Classification And Regression Tree（CART）
ID3 训练基于：
Entropy 熵：
$$E(x) = -\sum_{i=1}^np_ilog_2p_i$$
这里的 $p_i$ 表示某一个成分所占比例；熵表示的是系统的不确定程度，越大越不确定
Information Gain：
$$Gain = E(U)-(P(u_1)E(u_1)+P(u_2)E(u_2))$$
这里的 E 表示 Entropy；  
U 表示分割前的集合；  
$u_1、u_2$ 表示分割后的集合；  
$P(u_1)、P(u_1)$ 表示分割后子集占的比例，他们和为 1

#GBDT（Gradient Boost Decision Tree）

#Adaptive Boost
典型的提升方法。我之前的人脸检测项目在 Cascade 结构中使用过

