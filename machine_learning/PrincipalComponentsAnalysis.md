#主成分分析法（Principal Components Analysis）
这是一种常用的 降维 dimension reduction 数据预处理方法：
- 提取原始数据中的关键信息。在信号处理中，认为信号具有较大的方差，噪声有较小的方差，信噪比就是信号与噪声的方差比，越大越好；而主成分是原始数据中最大方差的维度，根据主成分进行降维可以去除噪声，提取关键信息。
- 进行数据压缩。这一方法最大程度保存数据信息，无论选择压缩到什么维度（保留多少主成分），剩下的主成分以及原始数据在其上的映射构成原始数据在这一维度下的最优矩阵分解 Matrix Factorization。

这一方法通常用于原始，是一种  的方法。这可以对原始数据进行压缩。

并且，其主成分构造的是而所以这一方法可以帮助我们从复杂的原始数据中提取主要信息。

##主成分（Principal Components）定义
主成分是指，对于原始数据矩阵 $X$ 方差最大化的一组投影矢量。设 $X$ 的协方差矩阵（covariance matrix）为 $C$，主成分为 $u$。

**这是一个 约束优化问题（constrained optimization problem）：**  
###$\max \limits_{u} u^tCu,\ \ subject\ to\ u^tu=1$  

- $u^tCu$ 是原始数据 $X$ 在$u$投影后方差值
- $u^tu=1$ 约束了投影向量的大小，使得我们在约束大小的情况下发现最优方向

这里利用 拉格朗日乘子（Laplacian Multiplier） 来求解主成分：
$L(u,\lambda)=u^tCu-\lambda(u^tu-1)\\
\displaystyle\frac{\partial L}{\partial u}=2Cu-2\lambda u=0\\
Cu=\lambda u$  

- 满足上式的向量 $u$ 被称作特征向量，相关的 $\lambda$ 被称为特征值
- 特征向量 $u$ 相互正交（orthogonal）
- 特征值 $\lambda$ 表示原始数据在相关特征向量上映射后的方差：$u^tCu=u^t\lambda u=\lambda$
- 所有特征值不为 0 的矩阵称为满秩矩阵

特征值 $\lambda$（eigenvalues）最大的一组特征向量 $u$（eigenvectors）称为 主成分（principal componentes）

##主成分 求解
设原始数据矩阵 $X \in R^{\ m\times n}$，$m$ 行 feature，$n$ 列不同数据  
相关协方差矩阵 $C = \displaystyle\frac{1}{N}\sum_n^N(x_n-\mu)(x_n-\mu)^t$, $\mu$ 为平均数据  
$C \in R^{\ m\times m}$ 的特征向量和特征值可由下述 Matlab 代码求出，后两行代码将升序排列转为降序排列
```
[eigen_vectors, eigen_values] = eig(C);
eigen_vectors = flip(eigen_vectors, 2);
eigen_values = flip(diag(eigen_values));
```
特征值的大小视为相关主成分能量大小，我们只选择少数靠前的特征向量作为主成分；例如能量总和占总能量90%的特征向量，这往往远远小于 $m$，所以能实现数据的大幅压缩且尽量保存信息

##主成分 的对偶求解
上述求解过程中，原始数据矩阵为 $X \in R^{\ m\times n}$，相关协方差矩阵为 $C \in R^{\ m\times m}$   
当数据维度 $m$ 远远大于数据样本数量 $n$ 时，$C$ 将会变成一个巨大的方阵而难以求解，图片数据处理中往往会出现这个问题（$100\times 100$  的二维图片 flatten 后将产生 $1\times 10^4$ 维度的特征）

这样的情况下，我们不会直接处理协方差矩阵 $C \in R^{\ m\times m}$，而是从它的对偶矩阵 $D \in R^{\ n\times n}$ 开始处理，$D$ 的规模远远小于 $C$，但是却可以等价地求出主成分！  
$A = \displaystyle\frac{1}{\sqrt{N}}(x_1-\mu,\ x_2-\mu,\ ...\ x_n-\mu)\\
C = \displaystyle\frac{1}{N}\sum_n^N(x_n-\mu)(x_n-\mu)^t = AA^t\\
D = A^tA$  
**我们假设 $u$ 是 $C$ 的特征向量，$v$ 是 $D$ 的特征向量，则有：**  
$\begin{aligned}Dv &= \lambda v\\ 
A^tAv &= \lambda v\\
AA^tAv &= \lambda Av\\
CAv &= \lambda Av\\
\end{aligned}$  
又由于 $Cu = \lambda u$，则  
$u = Av$  

这就建立了协方差矩阵 $C$ 与其对偶矩阵 $D$ 的直接关系，当数据维度 $m$ 远远大于数据样本数量 $n$ 时，我们应该在对偶空间求出 $D$ 的特征向量和特征矩阵，再转换到 $C$ 的特征向量；注意，这俩个矩阵的特征值是完全一致的

最后要注意的一点是，v 是协方差矩阵直接分解的结果，是标准化的，其二范数为 1；但是 u 是我们的转换结果，必须在进行一次 normalize，保证特征向量二范数（magnitude，$uu^t$）为 1： 

$u = u \ /\ ||u||_2 $ 或者 $ u = u\ /\sqrt{\lambda}$

##SVD（Singular Value Decomposition） 求解
奇异值分解是常用的求取主成分的方法，其公式为：$X=U\Sigma V^T$

- $U \in R^{m \times m}$ 被称为左矩阵，它由 $XX^t$ 的特征向量组成
- $V \in R^{n \times n}$ 被称为右矩阵，它由 $X^tX$ 的特征向量组成
- $\Sigma \in R^{m \times n}$  被称为奇异值矩阵，它由上述俩矩阵的特征值组成（俩组特征向量对应一致的特征值）


SVD可以同时求出原始矩阵俩个方向上的特征向量


##PCA 基本流程
- 数据 flatten，例如将二维图片拉伸为一维向量
- 求取成分（特征向量），可以直接构造协方差矩阵求解，也可以根据情况使用对偶求解或者SVD
- 由特征值（即数据映射后的能量）筛选出主成分，例如筛选出最重要的包含90%能量的特征向量
- 将原始数据逐个投射到主成分空间。一般，我们会先把原始数据矩阵 $X$ 零均值化（逐列减去平均数据），再乘以主成分矩阵 $U$ 来完成投射操作：$Z = (X-\mu)^tU$
- 最后，我们可以通过主成分空间中的压缩数据 $Z$ 重建原始数据 $\hat{X}$：  
$\hat{X} = \mu + Z*U^t$
这样我们可以对比 $X$ 和 $\hat{X}$ 来验证我们的算法实现是否正确、合理

##其他常用特征处理
###线性判别式分析 LDA（Linear Discriminant Analysis）or FLD（Fisher Linear Discriminant）
Fisher Ratio：$E = \displaystyle\frac{(\omega^tm_1-\omega^tm_2)^2}{\omega^tC_1\omega+\omega^tC_2\omega}$，$\omega$为投影方向

###独立成分分析 ICA（Independent Component Analysis）
假设信号由多个独立信号源产生，这一方法可以将混合信号解离。一个经典的例子是鸡尾酒宴会 cocktail 问题；我们有 m 个放在不同位置的麦克风记录到 n 个信号，实际的信号源（说话的人）数量少于 m。在知道独立源数量的情况下，我们可以通过 ICA 还原出每个人说的话。由于麦克风数量可以远大于独立源数量，这个方法通常还会使用 PCA 先对数据进行预处理，压缩维度到独立源数量，拿到主要信息后再进行信号解离。

###非负矩阵分解 NMF（Non-negative Matrix Factorization）
这一方法也是发现一组新的基来重新代表数据；但是这一方法着眼于 局部特征 local patterns 的发现，用组合局部模式的方法来重新代表数据集。经典的例子为将人脸分解为不同眼睛、鼻子、脸型等的组合，而非 PCA 的全局特征的叠加。这一方法常用于特征学习（重新组织特征），推荐系统（识别、细分用户群，商品类别），文档分类（TFIDF，Term Frequency Inverse Document Frequency 数据的稀疏矩阵分解）

###Autoencoder
####特征抽象
深度学习常用技术，无监督学习，特征预处理，每一层训练时，将输入（也就是前一层次的输出）通过 encoder 编码为 code，再由 decoder 解码还原，比对输入进行逼近。这一过程其实构造一个俩层神经网络即可实现，隐层表示编码的 code 空间，节点数少于输入层；而输出层是解码后的结果，节点数等于输入层，直接和输入层比对，得到残差。如果这一神经网络的激活函数是线性的，其最优解将是 PCA 的特征降维结果
####逐层抽象，构建深度学习
上面说到的编码结果 code ，相当于一个新的抽象层次来重新表征之前一个层次的输出，这是我们需要结果。而 decoder 在训练完之后就没有用了，只有 encoder 应用在最终模型中。我们以 code 作为这个层次的输出，继续训练下一次层级，这种逐层抽象化特征的方法和 RBM 一致；而这俩者的深度网络结构训练完成之后，不一定像 CNN 那样继续连接到 全连接网络（Fully Connected Neural Network）中，而可以作为特征接入到一般的分类器，如 Logistic Regression 或 SVM

###RBM
深度学习常用技术，无监督学习，特征预处理，这个是基于能量的角度进行了，目前我的理解，这也是 auto-encoder 的一种方式，常用于深度信任网络（Deep Believe Network）的预训练环节





