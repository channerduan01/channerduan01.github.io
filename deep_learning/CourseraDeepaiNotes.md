deepai相关笔记


# 深度学习为何突然崛起
尽管神经网络已经有数十年历史，但是基于神经网络的深度学习是最近几年才突然崛起的，主要有以下三方面的原因：

- Data: 大量的数据（labeled）采集，世界越来越信息化
- Computation: 计算能力增强，大规模机器学习（分布式计算），GPU以及更新的硬件设备
- Algorithms: 更有效的算法，例如 RELU 取代 Sigmoid，大幅减少了计算量（导数是常数）

# 研究人员的工作形式
正循环如此：Idea -> Code -> Experiment -> Idea ...
通过不断地迭代来优化我们队业务、模型的认知，逐渐打造出合适的模型（神经网络结构、各种参数、特征...）
迭代速度越快，这一循环模式走的越快，所以 Computation 和 Algorithms 效率上的提高有重大意义。


# Logistic Regression as A Simple Example

## loss function & cost function are two different terms
The loss function computes the error for a single training example, but the cost function is the average of the loss functions of the entire training set.

## Computation Graph
前向传播 Forward Propagation 和 后向传播 Backward Propagation 都可以看做是在 Computation Graph 上的计算。
BP 算法通过在 Computation Graph 使用 链式求导 Chain Rule 展开是非常清晰的（just step by step）。

## Vectorization
**Whenever possible, avoid explicit for-loops (besides, using built-in function to avoid for-loops)**
计算向量化对于 Deep Learning 这是重要技巧，因为计算量实在太大，所以需要特别注意工程实现细节。目标就是把各种 for loop 转化为向量化的矩阵乘法。向量化代码在底层执行时候能得到很大的并发优势（例如 SIMD 加速），这点无论 CPU 或是 GPU（GPU 潜力更大一些）都存在。

## Broadcasting in Python
在 Python 中，俩个 np.array 直接使用符号 '*' 相乘表示 element-wise 相乘；而真正的矩阵相乘需要使用 np.dot(a,b) 才可以。对于 element-wise 乘法，row 和 column 不满足的矩阵会被 Broadcasting 机制自动补全；例如 a.shape 为 (m,n)，对于 a*b：

- 若 b 为单个数，则补全为 (m,n)
- 若 b 为 (1,n)，则按行叠加 m 行补全为 (m,n)
- 若 b 为 (m,1)，则按列叠加 n 列补全为 (m,n)

然后再进行 element-wise 乘法。注意对于 (m,) 这类的 rank-1 array，计算会更trick，直接不要使用（通过 reshape 函数转为正常 matrix 或 vector）。


# Nerual Network

## Layer 计数
只有1个隐层的网络，称为2层网络，我们只计数 hidden layer 和 output layer，因为 input layer 没有相对应参数。
Logistic Regression 可以看作是一个只有一个 output layer 的 shallow model。

## Why deep representations?
- 帮助 catch 到不同层次的特征。经典的 CNN 的从边缘特征，到形状特征，到人脸特征
- 深度网络能更方便地描述一些复杂函数。 


## 各个节点的参数（W、b）初始化为随机
这里其实包含两个点：
- 要主要随机初始化！如果参数都初始化为 0，那么所以同一层次下的所有 node 都会 forward 输出同样的值，然后 backward 收到同样的梯度，最后所有 node 都卡在同样的参数上，实际上就把神经网络直接退化了~
- 一般初始化为较小的值！如果使用 activate function 为 sigmoid，如果初始化非常大的 weight，就会导致节点的输出特别大或者特别小，然后 sigmoid 上的梯度变成一个极小值，阻碍模型学习。所以一个常见套路是 ”rand(1)*0.01“。





