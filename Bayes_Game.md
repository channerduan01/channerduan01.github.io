#Game Show about Bayes Theorem
本文简单讲述一个有趣的经典的 Bayes定理 相关的小故事，展示了后验概率 posterior probability 出乎意料的威力。   
##Question
我们有三个杯子倒置桌上 $g1, g2, g3$，其中一个杯子下隐藏有 candy。当我们选择一个杯子后，剩下俩个杯子中的一个空杯子将被移除；此时我们是否应该变更选择，以更大概率拿到 candy？  
##Intuition
简单来说，我们的首次选择仅仅依靠先验概率 prior probability，有 $\displaystyle\frac{1}{3}$ 的概率拿到 candy；如果一个空杯被移除后，我们坚持选择，概率依然是 candy。然后，如果我们改变选择，选择移除后留下来的那个杯子，概率将变为 $1-\displaystyle\frac{1}{3} = \displaystyle\frac{2}{3}$，整整翻了一倍。  

貌似不相关的事件，却极大地影响了最终结果。上述移除操作本身，和我们的任务目标是耦合的，移除操作带来了额外的信息；如果移除操作是剩下的俩个杯子中随机移除一个的话，概率将不会发生变化。如果一尘不变地抱持着先验经验不放，不对事态的发展做出适应或调整的话，可能会出乎意料地产生误判，错失机会。
##Mathematics
###我们首先对游戏规则进行量化表述：  
对于三个杯子 $g1, g2, g3$，  
candy 存在于各个杯子的先验概率相同，设为变量 C：$p(C=1) = p(C=2) = p(C=3) = \displaystyle\frac{1}{3}$  
假设我们已经选择了：$g1$ ($g2, g3$ 也是类似的)   
移除杯子事件设为变量 R (remove) ：$p(R=1),\ p(R=2),\ p(R=3)$
###可以得到 remove 规则，条件概率分布 P(R|C) 如下： 

|          | R=1      | R=2      | R=3      |
|----------|----------|----------|----------|
| C=1      | 0        | 1/2      | 1/2      |
| C=2      | 0        | 0        | 1        |
| C=3      | 0        | 1        | 0        |

$g1$ 已被选择，所以没有可能 remove。当目标 candy 存在于剩下的俩个杯子中时，remove 的必然是空杯子，必然留下存在 candy 的一个，概率为 1。分布 P(R|C) 是 remove 的规则，如果这个操作和 candy 的存在独立的话，上述表格中的每一行应该都是 0，0.5，0.5；这种耦合事件作为我们的已知观察，将会修正最开始的先验分布。这里我们展开基于这个 将会改变我们对于 candy 的先验分布。
###基于 remove 结果，修正目标概率评估，求取后验概率分布 P(C|R)
Obviously，remove 操作本身的先验概率如下 (R=3 类似 R=2)：
$$p(R=2)=\sum_{i=1}^3p(R=2|C=i)\ p(C=i)=\displaystyle\frac{1}{2}$$
$$p(C=1|R=2)=\displaystyle\frac{p(R=2|C=1)\ p(C=1)}{p(R=2)}=\displaystyle\frac{\displaystyle\frac{1}{2} * \displaystyle\frac{1}{3}}{\displaystyle\frac{1}{2}}=\displaystyle\frac{1}{3}$$
$$p(C=3|R=2)=\displaystyle\frac{p(R=2|C=3)\ p(C=3)}{p(R=2)}=\displaystyle\frac{1 * \displaystyle\frac{1}{3}}{\displaystyle\frac{1}{2}}=\displaystyle\frac{2}{3}$$

所以，不管 R=2 或者 R=3，移除后剩下的杯子，都拥有我们当前选择杯子两倍的命中几率。   
We should switch to the left one!








