#Introduction
神经网络乃是目前人工智能与机器学习最为热门的模型，从基础的 Perceptron，Back Propagation 发展到深度学习 Deep Learning including DBN (Deep Believe Network)，CNN (Convolutional Neural Network)，RNN (Recurrent Neural Network) and etc。这一以神经元信号传播方式为基础的模型具有强大的描述能力，广泛地应用于 Regression 和 Classification 问题，是现代机器学习的主流技术.本文将简要介绍其基础理论.

#Single-layer Perceptron and Multi-layer Perceptron (MLP)
感知器 Perceptron 是最简单的神经网络形式，相当于一个单一的神经元节点，其将所有输入加权相加并加入自己的偏差bias后，通过激活函数产生输出。单层感知器就是一个 Perceptron 而已，相当于神经网络只包含输出层，而 MLP 至少包含一个 hidden 层。  
*非常重要的一个事实：如果这俩种神经网络只使用线性激活函数，他们完全等价！无论多么庞大的多层网络结构将等价于一个单一神经元节点，可见激活函数何等重要！Multi-layer perceptron (MLP) with linear nodes is no more powerful than a single-layer perceptron!*

#激活函数 Activate Functions:
Essentially, 没有非线性激活函数的神经网络不是神经网络！其神经元前向传播 Forward Propagation 将等价于乘以权值矩阵；即使其隐层拥有无限节点，将原始特征投射到无限维度的空间上，这也仅仅包含旋转 rotate、平移 translation、缩放 scale 的线性变化操作。线性不可分的数据无法基于线性变化而可切分！

激活函数极其重要，其为模型加入了非线性因素，加强了整个模型的表达力。这和人类神经元机制类似，人类神经元存在着稀疏反馈的重要特性，一个特定事件只能激活、点燃 (activate、fire) 巨大神经网络中的一小部分结构。激活函数可以抑制、过滤掉多余的反馈路径，实现稀疏反馈。


#神经网络梯度公式
对于每一个网络内神经元，神经元梯度 Gradient: $\eta = error\ f^{\prime}(net)$, $error$表示这个神经元所有输出上的合计偏差，$net$ 表示这个神经元的总输入。




###Relu (Rectified Linear Unit)

#Regression and Classification
softmax

$x^{10}=2$

#Gradient Decsent:  

###Opertimizers  
BGD - Batch
SGD — Stochastic Gradient Descent


###Moment  


###Weight decay(regularization item)



#What is deep learning
gradient diffusion


#Neural Network & Deep Learning
初期的神经网络模型非常小，关键瓶颈在于训练过程中 Back Propagation 残差扩散 (Gradient diffuse)，深层次模型无法进行训练。后续的深度学习，采用很多方法解决这一问题；例如深度信任网络的预先训练，先利用无监督学习逐层预训练 (pre-trained) 整个神经网络，完成各个层级的特征提取，为各个层级都生成合适的初始化参数；最后再用 Back Propagation 进行监督学习来微调模型。深度学习的目标，在于通过复杂的层次结构，对原始数据构造出不同层次的抽象，每一个层次都是对上一个层次的进一步抽象；这极大限度地利用了原生数据的价值 (深度模型智能预处理数据)，并且符合人类对世界的逐层抽象的认知规律 (在人类视觉感知上已被证实)。当然，深度学习的明显缺陷就是模型复杂，需要大量数据和时间才能有效训练，大量的标记数据和运算资源是昂贵的成本。


#CNN
###Convolution Layer, Pool Layer, Fully Connected Layer

#Latest Methods:  
##Dropout