#Introduction
神经网络乃是目前人工智能与机器学习最为热门的模型，从基础的 Perceptron 感知器，Multilayer Perceptron 多层感知器，Back Propagation 发展到 Deep Learning 深度学习；深度学习的研究集中在了 DNN（Deep Neural Network）深度神经网络；典型结构上包含 DBN (Deep Believe Network)，CNN (Convolutional Neural Network)，RNN (Recurrent Neural Network) and etc.；这一以神经元信号传播为基础的模型具有 强大的描述能力，广泛地应用于 Regression 和 Classification 问题，是现代机器学习的主流技术；本文将简要介绍其基础理论。

#Single-layer Perceptron and Multi-layer Perceptron (MLP)
感知器 Perceptron 是最简单的神经网络形式，相当于一个单一的神经元节点，其将所有输入加权相加并加入自己的偏差bias后，通过激活函数产生输出。单层感知器就是一个 Perceptron 而已，相当于神经网络只包含输出层，而 MLP 至少包含一个 hidden 层。  
*非常重要的一个事实：如果这俩种神经网络只使用线性激活函数，他们完全等价！无论多么庞大的多层网络结构将等价于一个单一神经元节点，可见激活函数何等重要！Multi-layer perceptron (MLP) with linear nodes is no more powerful than a single-layer perceptron!*

#激活函数 Activate Functions:
Essentially, 没有非线性激活函数的神经网络不是神经网络！其神经元前向传播 Forward Propagation 将等价于乘以权值矩阵；即使其隐层拥有无限节点，将原始特征投射到无限维度的空间上，这也仅仅包含旋转 rotate、平移 translation、缩放 scale 的线性变化操作。线性不可分的数据无法基于线性变化而可切分！

激活函数极其重要，其为模型加入了非线性因素，加强了整个模型的表达力。这和人类神经元机制类似，人类神经元存在着稀疏反馈的重要特性，一个特定事件只能激活、点燃 (activate、fire) 巨大神经网络中的一小部分结构。激活函数可以抑制、过滤掉多余的反馈路径，实现稀疏反馈。  

常见的激活函数如下：  
###Relu (Rectified Linear Unit)
分段函数，$(-\infty,0)$ 为0，而 $[0,+\infty)$ 为线性的 $y=x$，注意其斜率为常数1哦
###Sigmoid（or Logistic）
将 $(-\infty,+\infty)$ 映射到 $(0,1)$
###Tanh 
类似 sigmoid，其将 $(-\infty,+\infty)$ 映射到 $(-1,+1)$

#神经网络梯度公式
对于每一个网络内神经元，神经元梯度 Gradient: $\eta = error\ f^{\prime}(net)$, $error$表示这个神经元所有输出上的合计偏差，$net$ 表示这个神经元的总输入。

#Gradient Decsent:  
####Opertimizers  
BGD - Batch Gradient Descent，批量数据集计算梯度（也有说此方法指使用全量数据，mini-batch gradient descent 使用固定数量数据）
SGD — Stochastic Gradient Descent，单个数据计算梯度

####Momentum  
动量；用于搭配 SGD 使用；SGD 随机性大，下降方向不稳定，所以加入 动量 因素，每一次的下降速度，将部分（看超参数设定）由上一次下降速度决定；下面实例中，mom 即为动量超参数，alpha 为学习率，grad 为梯度
```
        velocity = mom*velocity+alpha*grad; 
        theta = theta-velocity;
```
####Weight decay(Regularization item)
这是用于抗 overfitting 的，其实就是 Ridge Regression 加入的 ；每次迭代
####Annealing learning rate

#Neural Network & Deep Learning
初期的神经网络模型非常小，关键瓶颈在于训练过程中 Back Propagation 残差扩散 (Gradient diffuse)，深层次模型无法进行训练。后续的深度学习，采用很多方法解决这一问题；例如深度信任网络的预先训练，先利用无监督学习逐层预训练 (pre-trained) 整个神经网络，完成各个层级的特征提取，为各个层级都生成合适的初始化参数；最后再用 Back Propagation 进行监督学习来微调模型。深度学习的目标，在于通过复杂的层次结构，对原始数据构造出不同层次的抽象，每一个层次都是对上一个层次的进一步抽象；这极大限度地利用了原生数据的价值 (深度模型智能预处理数据)，并且符合人类对世界的逐层抽象的认知规律 (在人类视觉感知上已被证实)。当然，深度学习的明显缺陷就是模型复杂，需要大量数据和时间才能有效训练，大量的标记数据和运算资源是昂贵的成本。

#CNN（Convolution Neural Network）卷积神经网络
####Convolution Layer, Pool Layer, Fully Connected Layer

#RNN（Recurrent Neural Network）循环神经网络

####LSTM（Long Short Term Memory）长短时记忆模型

#Deep Belief Network

####Autoencoder

####Restricted Boltzmann Machine

####Sparse Coding
 
####Dropout

#Latest Tech

####AlexNet，取得历史性突破
AlexNet 是一种典型的 convolutional neural network，它由5层 convolutional layer，2层 fully connected layer，和输出层 label layer (1000个node, 每个node代表ImageNet中的一个类别) 组成；2012年 Alex Krizhevsky 设计了这个8层的CNN刷新了ImageNet的image classification成绩，引起了computer vision community 的强烈关注。这篇文章的出现也是 deep learning 开始被 computer vision community 接受的关键转折点。
####VGG-Net
VGG-Net同样也是一种CNN，它来自 Andrew Zisserman 教授的组 (Oxford)，VGG-Net 在2014年的 ILSVRC localization and classification 两个问题上分别取得了第一名和第二名，VGG-Net不同于AlexNet的地方是：VGG-Net使用更多的层，通常有16－19层，而AlexNet只有8层。另外一个不同的地方是：VGG-Net的所有 convolutional layer 使用同样大小的 convolutional filter，大小为 3 x 3。
####GoogleNet

#神经网络加速的4个层次
这都是从各个层次寻找平衡，达到业务目标的过程
####数据层
对数据预处理，例如数据降采样；这牺牲了数据信息表达度，也是牺牲了后续的模型精度
####模型层
简化训练模型，牺牲模型精度；例如削减神经网络节点数，或者增加卷几层 stride
####算法层
定制化算法，底层篡改 mxnet 等模型的实现细节，牺牲 generalization 来换取对于特定问题、特定场景的性能优势
####硬件层
定制化硬件

#相关重要会议
####计算机视觉
CVPR(Computer Vision & Pattern Recognition)
####AI
AAAI
####机器学习

####深度学习
ICLR












