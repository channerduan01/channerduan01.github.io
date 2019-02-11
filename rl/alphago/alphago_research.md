# Alpha Zero
本文对 AlphaGo、Alpha Zero 相关算法思想进行基本整理，陈述。主要分为两部分，首先是介绍棋类游戏 AI 的通用性搜索技术 MCTS，然后根据实例展开介绍 Alpha Zero 相关算法细节和实现。

## 关键特性

Official link of it: https://deepmind.com/blog/alphago-zero-learning-scratch/

- No feature engineering
AlphaGo Zero only uses the black and white stones from the Go board as its input, whereas previous versions of AlphaGo included a small number of hand-engineered features.

- Only one network
It uses one neural network rather than two. Earlier versions of AlphaGo used a “policy network” to select the next move to play and a ”value network” to predict the winner of the game from each position. These are combined in AlphaGo Zero, allowing it to be trained and evaluated more efficiently.

- No rollouts for Monte Carlo Tree Search
AlphaGo Zero does not use “rollouts” - fast, random games used by other Go programs to predict which player will win from the current board position. Instead, it relies on its high quality neural networks to evaluate positions.


## 1. MCTS (Monte Carlo Tree Search) 蒙特卡罗树搜索
属于启发式搜索，是一种决策算法，搜索次数越多，结果越准确（逼近最优）；常见于棋类AI。
相关wiki：https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

### 搜索流程
其树结构是不平衡的，算法会关注 promising 的分支。其每个搜索回合分为4个步骤：

#### Selection
从根节点（当前局面的状态）开始，向下寻找一个最迫切（这是 MCTS 的核心，通常使用 UCB 公式来评估，均衡 explore & exploit）评估的局面。
- 如果找到一个已经完全展开的节点，就迭代地向下 selection，往树的分支上走下去
- 如果找到一个结束节点，直接转到 backpropagation
- 否则，进行下一步 expansion

#### Expansion
对当前节点进行展开（选择一个尚未展开的动作），生成一个子节点；这个新生成的节点需要接下来的 simulation 对其进行评价。

#### Simulation(也称为 rollout)
从待评估节点开始，随机地（或者以更好的 policy，例如 AlphaGo 的快速走棋网络）推演接下来的局面变化，得到一个最终评分（0/1表示是否获胜）。
这里随机搜索次数越多，评估的越准确。

#### Backpropagation
将评估的分数沿着搜索树路径逐级上传，更新路径节点的价值分数（直观说就是各个节点上的胜率更新了，探索次数也增加了）。

### 节点评估
节点的打分直接影响 selection 过程，决定了搜索树如何生长，这是 MCTS 和核心。
评估思想类似 Multiarmed bandit problem 中寻找一个合适的 arm，常使用 UCB (Upper Confidence Bounds) 来平衡 explore & exploit 问题（采用 UCB 的 MCT 又被称为 UCT = MCTS + UCB ）。

UCB 公式：$v_i+C*\sqrt{lnN/n_i}$，让算法关注胜率更高，或者结论不太确定的部分。

- $v_i$ 是当前节点的价值（例如使用胜率 $win_i/n_i$）
- $n_i$ 是当前节点的访问次数
- $N$ 是当前节点父节点的访问次数
- $C$ 是正整数超参数，调整 explore 的权重

这里还有个细节：搜索树其实在做 adversarial search，常见的下棋场景上，不同执棋方的价值是相反的， $v_i$ 需要适配不同执棋方。

### 实现上的细节

- 搜索算法可以在任何时间终止，返回当前最优结果。（如果无限地搜索下去，应该会找到真正的最优解）
- 搜索树中设计的状态，可以全部cache住，持续使用。（当真正走棋之后，搜索树顺着相关分支下移）

## 2. Alpha Zero 相关 MCTS 实现

### 搜索流程
在某一个待决策的局面下，在有限时间内，进行 n 次 MCTS 搜索（例如1百万次），获取最有利的一步棋。这就是 ALpha Zero 下棋时候需要做的全部工作。
这里的 MCTS 代码实现上非常简洁，每次搜索（又称 playout）严格分为下面 4 个步骤顺序执行一次：

- Selection：从当前局面根节点开始，递归向下 greedily 行走到叶子节点
- Expansion：如果这个选中的叶子节点不是终局（胜、负、平），那么对其所有可能的子局面进行展开，全部添加到子节点上；并且，通过高性能 NN 模型对所有 action 产出一个 [0,1] 的 prior 打分预判，作为相关子局面的 prior 打分（对于 MCTS_pure 则使用均值作为 prior，即平均分布）。
- Simulation：上一步的 NN 模型对当前局面所有 action 打分的同时，也会产出对当前局面本身的打分（对于 MCTS_pure 使用传统 rollout 随机走棋产出局面本身打分）。这里注意一个细节，如果当前局面已经是终局，则会直接使用 胜=+1/负=-1/平=0 的真实局面分数。
- Backpropagation：迭代更新当前局面所有父节点的分数

### 节点评估
这里使用了类似 UCB 的公式，最大不同是引入了 prior 来直接影响下一步局面的展开方向。
$Score = q + C*prior*\sqrt{N/1+n}$

- $q$ 是当前节点的价值（局面本身打分）；初始值为 0 ，是使用公式：$q = q + (value_{leaf} - q)/n$ 来更新的（即每次相关子节点搜索结束后的 Backprogagtion 过程，$value_{leaf}$ 是子节点的局面价值）
- $n$ 是当前节点的访问次数
- $N$ 是当前节点父节点的访问次数
- $C$ 是正整数超参数，调整 prior & explore 的权重

### 决策
最终决策没有使用节点打分，而是使用访问次数，这点有点奇怪～ 另外，不同走棋模式下，决策稍有不同。

- MCTS_pure 纯粹 greedily 选择 visit 次数最多节点；每次走棋完重置搜索树
- Alpha Zero 下棋模式使用了一定采样（$softmax(1.0/1e^{-3} * np.log(np.array(visit) + 1e^{-10}))$），但依然近似对 visit 做 greedily；每次走棋完重置搜索树
- Alpha Zero 学习模式上会在上述采样基础上，加入一定噪声来帮助 explore；每次走棋完复用搜索树


## 3. Alpha Zero 相关 nn-structure
### input
网络结构上，使用原始棋局的 0/1 二值化局面直接输入，基本上没有其他先验知识和特征工程。
Alpha Zero 使用了 17 个 19*19 的平面来描述局面：
- 2*8个平面描述双方最近8步的棋子情况
- 最后1个平面描述是否先手；先手则整个平面全为 1，否则全为 0

### nn
网络结构上使用了卷积神经网络来更好地提取局面特征，并且使用了 Multiple Task Learning，同时训练 策略模型（局面搜索剪枝） 和 价值模型（局面价值判断）。主要分为以下三个部分：

- 公共特征提取；整个局面先接入多层卷积网络进行特征提取，Alpha Zero 连续使用数十层卷积。我们的样例程序中使用3层（kennel 3*3）来简单模拟，从 4-channel 的原始输入，提升维度到 128-channel，作为对局面的基本特征提取。（这里相对奇怪的点是：没有使用 pooling，卷积后一直保持着局面的长宽尺寸）
- policy network 策略网络；对卷积后特征使用 kernel 1*1 降维（样例中降到 4-channel），然后拉平并通过全链接网络输出整个局面大小的 logits（样例中直接用 log_softmax 产出 log 概率），即表征下一步各位置的下棋概率。
- evaluation network 价值网络；对卷积后特征同样降维（样例中降到 2-channel），然后拉平并通过 ->64->1 2级全链接网络输出单个值，使用 tanh 变化到 (-1,1) 和 label 的对局 负=-1/胜=+1 对应，即表征当前局面的价值。

### 损失函数
$$loss = loss_{evaluate}+loss_{policy}+regularizer\\
= (z-v)^2-\pi^T\log{p}+c||\theta||^2$$

loss 分三部分构成，evaluation 和 policy 两部分的误差可以看作是同权重的 Multiple Task Learning：

- $\pi$ 是 policy label，取值为 [0,1]，是对局过程中，由 MCTS 决策的下一步走棋概率
- $z$ 是 evaluation label，取值为 -1/+1/0，表示最终对局结果
- $\theta$ 是模型所有参数（剔除 bias）做正则化（$c$ 为超参数）

### 模型输出

- train：训练过程中，输出按照上述流程输出整个下一步的 policy 概率分布和当前棋局评估 score
- infer：推断过程中，唯一不同的是：会对 policy 结果做一下裁剪，干掉无效棋位（已经被占据）的概率输出


## 4. Alpha Zero 相关 训练细节

### The key: self-learning
Alpha Zero 强调 0 先验知识，这里面关键的一点，就是模型不会输入任何其他人的对局过程进行训练，完全只依靠自己左右互搏的经验来学习。在自己对抗自己的过程中，模型越来越强，产出的新的数据也越来越来价值，整体迭代提升。

样例实现中，会使用当前待训练模型自己与自己对抗，生成训练数据 $(s,\pi,z)$：

- $s$ 为当前局面，实际上就是待输入卷积神经网络的 size=(4,width,height) 的 0/1 二值化局面；对应模型特征
- $pi$ 为当前局面 MCTS 搜索后，实际决策时的走棋概率分布；对应模型策略网络label
- $z$ 为当前局面下棋方，最后是否获胜 取值为 -1/+1/0；对应模型价值网络label

每一次 train 迭代结束，就会展开一轮自我对抗，生成大约数十个 pair 的训练数据（通过等价局面旋转、对称继续扩充到上百个），加入到 memory 中，然后再从 memory 中随机 sample 出一个 batch 的训练数据来 train。样例中 memory_size=10000（是一个deque，新数据加入会把最旧数据清除）、batch_size=512。

这里自我对抗的一个关键点是：模型是进入学习模型，引入更多随机、噪声因素，大大强化 explore，以探索更好的棋路。

### 模型收敛监控
强化学习的模型训练过程必须严格把关，确保模型处于收敛过程中。这里有几个关键关键指标：

- 训练 loss 收敛减小；这个是最基础的维度
- 策略网络 KL 散度变化；每一次 train 迭代，都监控新、旧版本策略网络打分之间的 KL 散度，如果两者分布差异较大，则说明模型收敛有问题，需要自适应地降低学习率；反之则提高学习率。样例代码中使用了这种策略来动态适配学习率。
- 策略网络 entropy 变化；模型对于下一步的走棋应该越来越有信心，相关分布的 entropy 应该越来越小。
- 价值网络相对误差的变化；这里样例中使用了“可解释方差”指标（有点类似可决系数），公式为：$var_{explained}=1-\frac{D(Y|X)}{D(Y)}=1-\frac{var(y_{label}-y_{predict})}{var(y_{label})}$；这是一个 [0,1] 的值，越大越好（说明分布更接近label），每次 train 迭代都监控新、旧版本之间差异，新版本应该要大于老版本才对。

### 模型阶段评估
当模型训练到一定程度，有一些相对较公正、准确的评估方法：

- 对战大规模搜索的 MCTS；使用一些成本相对大的手段进行评估，例如样例中的模型只发起 400 次 MCTS 搜索，但是评估对战的纯随机 MCTS 版本发起 4000 次搜索；可以观察模型训练什么时候能够击败这种纯策略的搜索算法（可以持续增加策略版本的搜索次数来提高对战难度）。根据多轮对战，保留胜率最高的模型。

- 对战其他 Alpha GO 版本；尝试和已有的优秀模型版本对弈。


















