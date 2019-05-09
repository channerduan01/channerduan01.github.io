# 统计模型

马尔科夫模型是一种统计模型，是纯粹的概率统计，其广泛应用在语音识别、音(汉语拼音)字转换、机器翻译、词性标注等领域。

### 随机过程 random process
相比仅考虑分布的 random variable 随机变量，random process 随机过程更复杂。它研究的是系统随时间变化的整个过程，观测到的结果为一个序列而非单个值；其中每一个时刻对应到随机变量的分布。

### Markov Property 马尔科夫性质
当一个随机过程在给定现在状态以及所有过去状态情况，其未来状态的条件概率分布仅依赖于当前状态，即 $P(future|current,past)=P(future|current)$，那么此随机过程具有马尔科夫性质。具有**马尔科夫性质**的过程通常称为**马尔科夫过程**，时间和状态都是离散的**马尔科夫过程**称为**马尔科夫链**。

### 马尔科夫模型 (Markov Model)
马尔科夫链 是最简单的 马尔科夫模型，它建模系统状态以一个随机变量的形式随时间变化，且每一次状态改变（即下一个状态的概率分布）只和系统当前状态有关而和系统历史状态无关。实际中模型可以进行拓展，即 $P(s_{a+1}|s_{a},s_{a-1},...,s_{0})=P(s_{a+1}|s_{a},s_{a-1},...,s_{a-n+1})$ 下一个状态由前 n 个状态决定，称为 n 阶马尔科夫模型。当然，大多数情况随机过程分析中我们用的是经典的1阶模型。

### 时齐的马尔科夫链
即对于任意时刻，两个状态 $s_i$、$s_j$ 之间的转移概率是一致的，所以我们可以得到一个确定的状态转移矩阵 $A=[a_{ij}]_{|S|*|S|}$，位置$a_{i,j}$ 表示 $s_j$ 到 $s_i$ 的转移概率。大多数分析针对的都是时齐的马尔科夫链，否则不同时间将对应不同的状态转移矩阵，就太复杂了。
这里还涉及**马尔科夫的平稳分布**，对于某些符合条件的转移矩阵P，任意初始状态分布 $\pi=s_0$ 经过一定次数的转移后，都会收敛到一个确定的分布 $s_x$。MCMC抽样方法就是以马尔科夫的平稳分布为基础的。

### 隐马尔科夫模型 HMM (Hidden Markov Model)
相对于时齐的马尔科夫链，隐马尔科夫模型中我们无法直接观测到状态 $s$，而是观测到到 $o$，这引入了一个观测概率矩阵 $B=[b_j(k)]_{|S|*|O|}$，其中 $b_j(k)=P(o_k|s_j)$ 是状态 $s_j$ 下观测到 $o_k$ 的概率。结合上之前提及的状态转移矩阵 $A$ 和初始状态分布 $\pi$，隐马尔科夫模型包含如下三要素 $(A,B,\pi)$。

隐马尔科夫模型应用中的三个问题：
- 概率计算。给定模型 $(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,...o_T)$，计算此观测序列出现的概率。可以使用前向算法。
- 学习问题（搞定模型参数）。已知观测序列 $O=(o_1,o_2,...o_T)$，使用极大似然估计的方法估计模型参数 $(A,B,\pi)$。（这个比较难弄...）可以使用 Baum-Welch 算法
- 预测问题（解码）。给定模型 $(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,...o_T)$，计算出最优可能的状态序列（即隐藏序列） $S=(s_1,s_2,...s_T)$。可以使用维特比算法。

