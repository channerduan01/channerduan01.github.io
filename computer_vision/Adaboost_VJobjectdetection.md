# Adaboost 与 Viola-Jones 物体检测框架

本文会详细介绍 Adaboost 这一 ensemble 模型（属于 boosting类别）；并且详细阐述其相关的重要应用 Viola-Jones object detection framework。

## 1. Adaboost (全称 Adaptive Boosting)
### 1.1 算法特点
#### boosting
这是一个集成算法，也是一个迭代算法。我们每一轮迭代寻找一个合适的 weak learner 集成到模型中（本质上是梯度下降），通过 N 轮的迭代来集成出一个强分类器，这是一个 boosting 提升过程。

#### adaptive
这涉及实现提升的细节。所有的样本都有自己的weight，初始化时一致。training时每一轮迭代，我们要更新所有样本的weight，模型正确判别的样本weight减小，而错误判别的样本weight增加。这就是 Adaboost 的核心了，这非常像一个我们人类逐渐学习的过程。

#### steepest decent with approximate functional gradient
数学上也可以解释为类似在函数空间上做最速梯度下降。每次迭代过程中，选择的 weak classifier 函数其实是当前负梯度方向，而对应的权值则是在此方向使 loss 减少最多的步长。这里使用的 loss function 是 exponential function，而 GBDT（gradient boost decision tree）推广到了其他 error function，也可以说 AdaBoost 是一种 GBDT。

### 1.2 训练流程详解
实际上不同 Adaboost 版本的 $\epsilon$ 和 更新策略略有不同，这里讲解的是 VJ 论文中使用的版本。
#### 1.2.1 **init**
假设我们有 N 个样本 $D=[(x_1,y_1),(x_2,y_2),...,(x_n,y_n)]$，其中 $x$ 为特征向量，**$y$ 为 0/1 分类的 label**。
我们要迭代训练 T 轮，样本集 weight 向量 $u^{t=1,2...T}$ 初始化为 $u^{t=1}=[\frac{1}{N},\frac{1}{N},...,\frac{1}{N}]$，即 N 个样本的重要性一开始是一致的，且和为 1。
（这是正例负例等比例的情况，不等比例时，可以假设正例 M 个而负例 L 个则初始化为 $u^{t=1}=[\frac{1}{2M},\frac{1}{2M},...,\frac{1}{2L},\frac{1}{2L}]$，即保证正负例各自总权重皆为 $\frac{1}{2}$）

#### 1.2.2 **training iteration**
*（对于第 $t$ 轮迭代）*

- 选取 weak classifier 并计算 error：
根据当前样本权重 $u^{t}$ 获取一个当前最好的 weak classifier $h_t(x)$ 函数（训练一个 decision tree 或者 decision stump），函数输出 0/1 结果，其相关 error 为 $\epsilon_t=\frac{1}{N}\sum_{i=1}^Nu_i^{t}|h_t(x_i)-y_i|$。这里应该有 $0\leq\epsilon_t<0.5$，因为我们要求 weak classifier 至少优于乱猜。

- 更新训练集样本权重：
针对 $\epsilon_t$，我们设置一个**核心变量 scaling factor $s_t=\displaystyle\frac{1-\epsilon_t}{\epsilon_t}$（$1<s_t<\infty$）**，并更新样本集 weight ：
$$
u^{t+1}_i=\begin{cases}
u^{t}_is_t& h_t(x_i)\ne y_i,\ incorrect\ case\\
u^{t}_i& h_t(x_i)= y_i,\ correct\ case
\end{cases}
$$
除了上述公式，之后还会对 $u^{t+1}_i$ 重新 normalize $u^{t+1}_i=\frac{u^{t+1}_i}{\sum^N_{j=1} u^{t+1}_j}$，保证和为 1。
**可以注意到这里的 adaptive 的机制：本轮迭代 $h_t(x)$ 错误分类的样本的 weight 会增大（scale up incorrect），正确分类样本的 weight 相应减小（scale down correct）**

- 确认此 weak classifer 权重：
会根据本轮 $h_t(x)$ 的表现给予它一个权重 $\alpha_t=ln\ s_t$（$0<\alpha_t<\infty$）；当 $\epsilon_t=0$ 时，$\alpha_t=\infty$ 即对于完美的 classifier 我们可以给到无穷大的权重；而当 $\epsilon_t=0.5$ 时，$\alpha_t=0$ 即对于乱猜的 classifier 直接不予集成。可见 error 越小的分类器权重越大。

#### 1.2.3 **aggregation**
历经 T 轮训练后，将 T 个 $h_t(x)$ 线性集成为 strong classifier。实际上集成的参数在迭代过程中已经决定了，这又称为 linearly on the fly with theoretical guarantee，涉及的理论验证之后会详述。
$$
C(x_i)=\begin{cases}
1& \sum_t^T\alpha_th_t(x_i)\ge\frac{1}{2}\sum_t^T\alpha_t\\
0& otherwise
\end{cases}
$$
$h_t(x_i)$ 是 weak classifier 的 0/1 投票，$\alpha_th_t(x_i)$ 则是加权投票；当所有 weak classifier 对样本的加权投票结果大于整体权值的 $\frac{1}{2}$ 时，strong classifier 判定样本为 positive，否则为 negative。

### 1.3 Weak Classifier
这里把训练流程中的 weak classifier 单独拿出来详细说明。
#### 1.3.1 Compare to Random Forest
Random Forest 也是 ensemble 集成算法，属于 bagging 类别，通过 bootstrap 来 uniformly aggregate 了一组等权重的 decision tree，然后通过投票给出整体结果。这是使用 classification（输出 0/1 结果） 或者 regression decision tree 都可以。

这个也是 weak classifier 集成为 strong classifier 的过程，但是集成思想和 Adaboost 不一致（bagging vs. boosting）。Adaboost 也可以使用 classification decision tree 作为 weak classifier，但更常用的是更 weaker 的 decision stump。这里还可以想象一下，我们如果给 Adaboost 一个 fully grown 的 decision tree，那么可能会有 $\epsilon=0，s=\infty$，训练就崩坏了~ 所以即使用 decision tree 也要做约束出弱的树，而不是像 Random Forest 那样 fully grown。

#### 1.3.2 Decision Stump
这个是一个弱弱的 weak classifier，类似只有 1 层的树，只剩一个树桩了。具体公式如下：
$$
h_{f,p,\theta}(x)=\begin{cases}
1& if\ p(f(x)-\theta)>0\\
0& otherwise
\end{cases}
$$
$x$ 为样本，$f(x)$ 为某一 feature 维度上的取值，$\theta$ 为 threshold，$p$ 用于控制 direction。总的来说，decision stump 即选择一个特定 feature 维度，使用简单 threshold 和 direction 来进行决策。无论多么复杂的数据集，只需要3个参数就完全定义了这个弱弱的 decision stump，我们对它的期待只是比随机好一点点，$\epsilon<\frac{1}{2}$ 就好。

#### 1.3.3 Adaboost 中使用 Decision Stump
Adaboost 常用 decision stump 作 weak classifier，即训练流程中寻找的 $h(x)$ 就是 decision stump。在样本集 N-samples 和 特征空间 M-dimensions 中结合当时样本 weight 学习到最优 $h(x)$ 其实是一个**搜索过程**，也可以看做是 feature selection；目标是最小化加权 error rate：（VJ 版本的 $u_i^t$ 保证和为 1，所以下式不需要再除以 $\sum_{i=1}^Nu_i^t$）
$$\min_{h}\epsilon_t=\sum_{i=1}^Nu_i^{t}|h_t(x_i)-y_i|$$

##### 具体应用时还有 2 个性能上的考量：

- 时间复杂度
为了优化训练时间，在迭代之前要对M个维度依次排序并 cache 结果，复杂度 $O(M\cdot N*logN)$。之后每次迭代只需要 $O(M\cdot N)$ 就可以找到当前轮 $t$ 最优 $h_t(x)$。
- 空间复杂度
cache 住的结果如果存在硬盘的话会极大降低搜索速度，我们希望全部放在内存，这里空间复杂度为 $O(M\cdot N)$。对于 VJ 原论文中 N=1万、M=16万，考虑 int 存储 feature value 则 $cost=4\times 16\times10^{8} B=6.4 GB$。考虑到扩充特征集、扩充数据集、存储结构效能等问题，其实内存要求是很严峻的。

### 1.4 理论支持（参考了 Hsuan-Tien Lin 机器学习技法课程）
Adaboost 凭什么能完成 weak classifiers to strong classifier？凭什么可以收敛？可以越来越好？我们可以从数学层面（steepest decent with approximate functional gradient）找到充分解释，本节详细阐述。
#### 1.4.1 公式及符号重新定义
之前“1.2 训练流程详解”使用的是 VJ 论文中原版公式。这里为了更清晰解释数学理论，会改动一些公式定义：

- **定义数据 label y 与 classifier $h(x)$ 输出为 -1/+1 而非之前的 0/1**，这可以简化很多公式
 - $y_ih_t(x_i)=1$ 即表明判定正确
 - 弱分类的 error rate 写作：$\epsilon_t=\displaystyle\frac{\sum_{i=1}^Nu_i^t\ [ y_i\neq h_t(x_i)]}{\sum_{i=1}^Nu_i^t}$，这里是标准的 rate 了
 - strong classifier 判别公式写作：$C(x_i)=sign(\sum_t^T\alpha_th_t(x_i))$（使用符号函数 sign 直接处理正负向加权投票之和，非常简洁）
 - 这里将上式中弱分类器加权投票和记作 $vote\ score=\sum_t^T\alpha_th_t(x_i)$，后续数学推导使用
- 改变 scaling factor $s$ 公式为 $s_t=\displaystyle\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}$，比之前版本多了一个 square root，取值范围不变，这是为了方便后续数学推导

- 样本权重 $u$ 公式更改为下述，这个改动也不大，也是为了方便后续数学推导（原公式只考虑 scale up，然后 normalize；现在加入 scale down 而去掉了麻烦的 normalize）：
$$
u^{t+1}_i=\begin{cases}
u^{t}_is_t& h_t(x_i)\ne y_i,\ incorrect\ case\\
u^{t}_i/s_t& h_t(x_i)= y_i,\ correct\ case
\end{cases}
$$
然后这里最妙的一步是直接把上式简化为： $u^{t+1}_i=u^t_is_t^{-y_ih_t(x_i)}$

#### 1.4.2 loss function of AdaBoost
根据 1.2 中公式 $\alpha_t=lns_t$ 可以有 $s_t=exp\{\alpha\}$  

进而可以对样本权值公式进一步变换：
$$u^{t+1}_i=u^t_is_t^{-y_ih_t(x_i)}=u^t_i(exp\{\alpha_t\})^{-y_ih_t(x_i)}=u^t_iexp\{-y_i\alpha_th_t(x_i)\}$$
我们假设所有样本权值训练前初始化为 $u^{0}_i=\frac{1}{N}$，则可以得到经过 $T$ 轮训练后的样本权值：
$$u^{T+1}_i=u^{0}_i\cdot\prod_{t=1}^Texp(-y_i\alpha_th_t(x_i))=\frac{1}{N}\cdot exp(-y_i\sum_{t=1}^T\alpha_th_t(x_i))$$
可以看到任意样本 $x_i$ 的最终权重与所有 weak classifier 整体 $vote\ score=\sum_{t=1}^T\alpha_th_t(x_i))$ 有关，即：
$$u^{T+1}_i\varpropto exp\{-y_i(vote\ score\ on\ x_i)\}$$
上式右侧实际上就是 exponential error：

- 整体 strong classifer 可以写作 $C(x_i)=sign(vote\ score\ on\ x_i)$，所以 $y_i(vote\ score\ on\ x_i)$ 的正负表示判定正确或错误
- $y_i(vote\ score\ on\ x_i)$ **越大**，表示判定越正确，对应的负指数值就**越小**
- $y_i(vote\ score\ on\ x_i)$ **越小**，表示判定越错误，对应的负指数值就**越大**

这实际上就是 Adaboost 的目标就是 exponential error 越来越小（最优化问题），也可以说是让所有样本的权值越来越小；最终 error/loss function 为所有样本的 exponential error 之和：
$$E_{ADA}=\sum_{i=1}^Nu^{T+1}_i=\frac{1}{N}\cdot \sum_{i=1}^Nexp(-y_i\sum_{t=1}^T\alpha_th_t(x_i))$$

#### 1.4.2 loss function 内在意涵
上述的 exponential error 实际上可以找到直观解释。先变化 strong classifier 判别公式如下：
$$C(x_i)=sign(\sum_t^T\alpha_th_t(x_i))=sign(\sum_t^Tw_t\phi_t(x_i))$$
我们可以把 $\alpha$ 看作线性模型中的 weight，而一系列的 $h(x)$ 看做是 SVM（support vector machine） 中一系列函数映射 $\phi(x)$，SVM 正是利用映射原始数据到其他空间进行线性分割的。结合 SVM 优化目标：
$$margin=\frac{y_i(w^{trans}\phi(x_i)+b)}{||w||}$$
比对上式，其实 AdaBoost 包含带符号的非正规化 margin：

$signed\ and\ unnormalized\ margin=y_i\sum_{t=1}^T\alpha_th_t(x_i)$

AdaBoost 的目标是让这个 margin 尽量为正数且最大化，也就是让其负指数尽量小于1且最小化，也就是我们之前得到的 error/loss function。
另外，因为使用了 exponential，分类错误的 penalty 呈指数增长 $(1, \infty)$；而一旦分类正确，则 penalty 只在 $(0, 1)$ 范围。这里有个问题是 AdaBoost 对噪声或脏数据敏感。

#### 1.4.3 Gradient Descent
#### 1.4.3.1 标准套路
Gradient Descent 标准套路是我要从当前位置（模型状态）走一小步，让 loss 最快的下降（减少）：
$$\min_{||v||=1}E(w_t+\eta v)\approx E(w_t)+\eta v\nabla E(w_t)$$

- $w_t$ 表示第 t 轮迭代时，当前的 weight vector
- $v$ 表示走的方向（向量，有 length 约束）
- $\eta$ 表示走的步长（标量，一般是一个较小的正数）

目标其实是找到一个让 loss 函数下降最快的方向 $v$，上式的约等于号是一阶泰勒展开，再加上其他数学工具可以证明最优的方向是负梯度方向 $v=-\nabla E(w_t)$。

#### 1.4.3.2 AdaBoost 加性模型套路
**AdaBoost 的套路很类似，核心是其通过新加入函数来完成梯度下降，这个函数对应最速下降的方向**，而函数的权重对应了下降的步长。AdaBoost 在第 t 轮迭代时，优化目标如下：
$$\min_{h_t}E_{ADA}=\frac{1}{N}\cdot \sum_{i=1}^Nexp\{-y_i(\sum_{\tau=1}^{t-1}\alpha_{\tau}h_{\tau}(x_i)+\eta h_t(x_i))\}$$
我们在原始 loss function 中新加入一项 $\eta h_t(x_i)$，就是说在第 t 轮的时候，我们想找一个新的函数 $h_t(x_i)$ 给出 loss 最速下降的方向（相当于标准套路中的向量 $v$），$\eta$ 则同样是走的步长，整体上就是想优化 loss 而已。以下先对上式进行化简：
$$\min_{h_t}E_{ADA}=\sum_{i=1}^N\frac{1}{N}exp\{-y_i(\sum_{\tau=1}^{t-1}\alpha_{\tau}h_{\tau}(x_i))\}exp\{-y_i\eta h_t(x_i)\}=\sum_{i=1}^Nu_i^texp\{-y_i\eta h_t(x_i)\}$$
然后我们把上式看作：$\sum_{i=1}^Nu_i^texp\{0+\gamma\},\ \gamma=-y_i\eta h_t(x_i)$，即把 $\gamma$ 作为自变量（这样原函数导数还是原函数~），然后作一阶泰勒展开（或者说对原式作麦克劳林展开）：
$$\begin{aligned}\min_{h_t}E_{ADA}
&\approx \sum_{i=1}^Nu_i^texp\{0\}+\sum_{i=1}^Nu_i^texp\{0\}(-y_i\eta h_t(x_i))\\
&=\sum_{i=1}^Nu_i^t-\sum_{i=1}^Nu_i^ty_i\eta h_t(x_i)\\
&=\sum_{i=1}^Nu_i^t+\eta\sum_{i=1}^Nu_i^t(-y_ih_t(x_i))
\end{aligned}$$
这里注意，在当前第 t 轮，上式左侧的 $\sum_{i=1}^Nu_i^t$ 是一个确认的常数，不能再优化了。上式右侧的步长 $\eta$ 我们先认为它是一个固定值（反正肯定是一个非负实数）；这里我们的优化目标变为：
**发现一个方向 $h(x)$，使得 $\displaystyle\sum_{i=1}^Nu_i^t(-y_ih_t(x_i))$ 最小**。

**这里特别重要的一点是，$h_t(x)$ 确定的是方向，它的 scale 理论上是没有意义，这种情况下我们一般会约束其 scale（例如标准 GBDT 中就会）。但是！在 AdaBoost 中，我们的 $h_t(x)$ 本身的输出只是 -1/1 状态，本身就没有 scale 的概念，所以不需要额外约束！**

对上述目标进一步变化：
$$\begin{aligned}\sum_{i=1}^Nu_i^t(-y_ih_t(x_i))
&=\sum_{i=1}^N-u_i^{t}\ [y_i=h_t(x_i)]+\sum_{i=1}^Nu_i^{t}\ [y_i\neq h_t(x_i)]\\
&=-\sum_{i=1}^Nu_i^t+\sum_{i=1}^N0\ [y_i=h_t(x_i)]+2\sum_{i=1}^Nu_i^{t}\ [ y_i\neq h_t(x_i)]\\
&=-\sum_{i=1}^Nu_i^t+2(\sum_{i=1}^Nu_i^{t})\cdot\epsilon_t
\end{aligned}$$
经过变换后，上式除了 $\epsilon_t$ 都是 constant，我们要优化的目标只有最右侧的 $\epsilon_t=\displaystyle\frac{\sum_{i=1}^Nu_i^t\ [ y_i\neq h_t(x_i)]}{\sum_{i=1}^Nu_i^t}$。

**Recap 一下，我们想找到一个方向 $h_t(x)$ 使 $\epsilon_t$ 最小**。这本身就是 AdaBoost 定义的 weak classifier $h_t(x)$ 的选取标准，一般是通过在特征空间上搜索选取 Decision Stump 实现，这一过程近似地获取到最速下降方向。

#### 1.4.4 Steepest Descent
我们在上一小节中已经确认了让 AdaBoost loss function $E_{ADA}$ 最速下降的方向是某一 $h_t(x)$，我们接下来需要考虑在这个方向要走多大的一步（$\eta$）？

对于 Gradient Descent 的标准套路，$\eta$ 会作为一个训练的超参数（学习率），可以尝试定义不同的值，然后通过 validation 选择合适的学习率；有时还会加入 annealing 退火机制或其他机制来动态调节学习率。但是在 AdaBoost 中，我们每走一步的代价是巨大的：

- 不能直接计算出负梯度方向，而是通过一个子过程寻找一个近似最优函数方向，耗费时间
- 需要在最终模型上多集成一个函数，这直接影响最终模型使用性能

所以，我们希望每走一步，都尽可能走最大的一步，尽量减少走的步数。上一小节，我们固定 $\eta$，通过 $\min_h \hat{E_{ADA}}$ 确定了 $h_t(x)$ 方向（approximate functional gradient）；现在，我们使用这个 $h_t(x)$，通过 $\min_{\eta}\hat{E_{ADA}}$ 来走出减少 loss 的最大的一步（greedily faster），这个套路被称为 Steepest Descent。以下进行最优 $\eta$ 的推导（注意这里使用了泰勒展开之前的准确 loss 公式）：
$$\begin{aligned}
\min_{\eta} \hat{E_{ADA}}
&=\sum_{i=1}^Nu_i^texp\{-y_i\eta h_t(x_i)\}\\
&=\sum_{i=1}^Nu_i^texp\{-\eta\}[y_i=h_t(x_i)]+\sum_{i=1}^Nu_i^texp\{\eta\}[y_i\neq h_t(x_i)]\\
&= (\sum_{i=1}^Nu_i^t) \cdot ((1-\epsilon_t)exp\{-\eta\}+\epsilon_texp\{\eta\})
\end{aligned}$$
由于上式只有一个未知数 $\eta$，所以直接求导 $\displaystyle\frac{\partial\hat{E_{ADA}}}{\partial\eta}=0$ 来解出最优值：
$$
\frac{\partial\hat{E_{ADA}}}{\partial\eta}=(\sum_{i=1}^Nu_i^t) \cdot (-(1-\epsilon_t)exp\{-\eta\}+\epsilon_texp\{\eta\})=0\\
-(1-\epsilon_t)exp\{-\eta\}+\epsilon_texp\{\eta\}=0\\
\frac{\epsilon_texp\{\eta\}}{exp\{-\eta\}}=(1-\epsilon_t)\\
exp\{\eta\}exp\{\eta\}=\frac{1-\epsilon_t}{\epsilon_t}\\
\eta=ln\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}
$$
上式即标准的 AdaBoost 计算 weak classifier 相关权重的公式 $\alpha_t=lns_t=ln\displaystyle\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}$！

现在可以完整地说：AdaBoost 每一次迭代，会选取一个函数 $h_t(x)$ 近似 loss 最速下降方向，并计算 $\alpha_t$ 作为当前最优下降步长，来集成这个新的 weak classifer；本质上是做梯度下降，这是 AdaBoost 的理论基础。

# 2. Viola-Jones object detection framework
AdaBoost 这一模型最典型的一个成功应用是就 VJ 在2000年论文中提出的 Viola-Joines 人脸检测/物体检测 实时处理框架，主要包含以下三个部分内容：

- Feature: Haar-based feature (wavelet)
- Learning algorithm: AdaBoost
- Attentional Cascade

其中涉及的 AdaBoost 算法，本文上一部门已经详细讲解，现在主要展开 Feature 和 Cascade architecture 这俩点，VJ 这俩个贡献极大地提升了算法性能，实现了实时图片检测。首先，先对传统检测流程做一下整体认识：

- 图片预处理
 - 降采样，原始大图size太大处理太慢，例如把 1000*1000 的原始图片降为 500*500
 - 像素 intensity 压缩，一般的图片是3通道（$256^3$）的RGB，我们会把它压缩到单通道的 intensity($256^1$)
- 特征预处理
这里会根据 Feature 相关思路，计算出待检测图片的 Integral Image，这是 VJ 框架的核心
- 滑窗检测
先以训练使用的图片尺寸如24*24为滑窗尺寸，在待检测图片上逐次划过所有位置（x轴+y轴全量遍历，当然 stride 可以设置大一点），截取所有可能的图片进行判断；并且会逐渐扩大滑窗尺寸（如每次扩大1.5倍）完成各个尺度下图片的检测。滑窗的存在把检测问题转换为了类似识别问题，但是这里识别的数量级巨大，500*500 的一张处理后待检测图片，轻轻松松包含10万个需要确认的截取
- 图片检测
滑窗过程中每一个截取都需要调用图片检测来确认结果，这里实际上利用了 A



## Feature
关于特征，原文使用的是 24*24 大小的图片，



there can be 162,336 Haar features, as used by the Viola–Jones object detection framework, in a 24×24 pixel image window


## Attentional Cascade


