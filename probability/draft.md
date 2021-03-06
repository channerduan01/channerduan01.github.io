# Game Show about Bayes Theorem
如果A事件发生的概率是1/3
如果B事件发生的概率是1/3
如果C事件发生的概率是1/3
如果D事件发生的概率是1/3

那么，同时发生三件事的概率是多少？

解答：
假设独立，那概率就是C(3,4)*(1/3)^3*(2/3)=8/81
（请注意排列组合的基本公式...）


# 一道贝叶斯思维考察题目
有一对夫妇连续生了6个小孩，都是男孩。试问他们的第8个小孩是男孩的概率是多大？答案开放，但请注意这是一个概率题目，不是脑经急转弯...

解答：
最简单的思路是：每次生小孩都当作一次独立试验，这样第8个小孩是男孩的概率是 0.5 不受之前事件的影响。
贝叶斯的思路是：依然把每次生小孩都当作一次独立试验，且生男孩的 prior 是 0.5。但是基于这对夫妇已经连续生了6个男孩的事实，我们要修正这个 prior，最典型的就是使用 beta 分布，通过已有的观测得到一个 新的prior，第8个小孩是男孩的概率等于这个 新的prior；显然这个 新的prior>0.5，但是大多少取决与我们初始设定的 beta分布，而 初始 beta分布 依据已有领域知识设定相关 a、b 参数。


那我等答案间隙再出个和编程相关的概率题：假设有个可以等概率生成0-4的函数Random5()，问我们应该如何调用它来写个Random7()和Random1000()，既一个能等概率生成“0～数字减1”的新函数

解答：
设已有随机数产生器 $a$ 均匀地生成 $[0, a-1]$；与推广为产生器 $b (b>a)$。
则级联 $a$ 可以找到 $\min_{n}{a^n>b}$；设 $l=floor(a^n/b)$，当 $a^n$ 产生的随机数大于 $bl$ 时需要重新产生，否则直接输出 $a^n\ mod\ b$ 即可；一次性产出结果概率为 $\frac{bl}{a^n}$


