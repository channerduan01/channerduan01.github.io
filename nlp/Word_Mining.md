# 新词发现
基于统计的经典算法（无监督学习）：基于已有的大量语料，枚举可能所有新词（原始字的顺序组合，限制长度为 n-gram），然后计算一些统计指标（主要是 凝固度、自由度）进行过滤，尽量准确地筛选出潜在的新词。产出的新词还是需要进行人工审核确认。

## 凝固度
这是一个类似互信息 Mutual Information 的概念，衡量两个字之间的相关性，是否经常“凝固”在一块儿出现，足够组成新的词。

### 2-gram 的凝固度计算
$$凝固度 = \frac{p(ab)}{p(a)p(b)}$$
使用两个字的联合概率与各自边缘概率乘积的比值，比值越高越发“凝固”。

- 如果 a、b 两个字只是碰巧走到一起，足够的语料上应该会统计出 $p(a)p(b)\approx p(ab)$，他们之间没有任何关联，$凝固度 \approx 1$；

- 如果 a、b 两个字极其相关，必定同时出现，应该统计有 $p(a)\approx p(ab)$，$凝固度 \approx 1/p(b)$，一般情况下远远大于1。

### 3-gram 的凝固度计算
$$凝固度 = min(\frac{p(abc)}{p(ab)p(c)},\frac{p(abc)}{p(a)p(bc)})$$
对于较长的组合，计算公式会较繁琐；分母仍然是固定的联合概率，但是分子需要在任意两个子串中，选择同时出现概率最大的组合，让凝固度最小。

对于 4-gram 的凝固 'abcd'，就需要考虑 ('a', 'bcd'), ('ab', 'cd'), ('abc', 'd') 3种子串组合。

另外注意：不同 n-gram 的凝固度数值很不一样，n-gram 字数越大，统计越不充分，越有可能偏高（所以应该设置更高阈值）。


## 自由度
这里统计的是候选词左、右侧的其他字词分布情况（计算信息熵），考察其上下文是否足够丰富（熵足够大，搭配足够不确定），一个足够独立的词应该会被应用在不同上下文中。

一些凝固度高的半词结果如 “俄罗”、“巧克”，可以通过自由度的阈值筛选过滤掉。当然，一些自由度高的短语组合如 “吃了一顿”、“看了一遍”，又需要使用凝固度来过滤掉。

### 计算
不管多长的候选词，我们看的都是其左、右侧的自由度，所以任意 n-gram 公式一致。

#### 熵的标准公式：
$$H(x)=-\sum (x\in X)P(x)log_2P(x)$$

#### 自由度：
$$自由度 = min(H(left), H(right))$$

分别求取候选词左、右侧分布的熵，选较小值为最终自由度。

- 如果某一侧的搭配是完全确定的（如统计 “俄罗” 右侧100%搭配 “斯”），熵为0，$自由度 = 0$
- 如果两侧都有非常多搭配，且概率平均，不确定性就很大，熵很大（接近上限 $0\le H \le log|X|$），自由度很大。

## 新词筛选
筛选主要涉及3个阈值：

- 词频：出现频率越高的组合，成词可能性越大
- 凝固度：越发凝固的组合，成词可能性越大
- 自由度：组合对应的Context越发自由，成词可能性越大

具体阈值选择，不同语料会有较大差异。
网络blog上介绍的经验公式（30M文本，字母 n 表示 n-gram 即相关新词包含的字数）：

- 词频>200, 凝固度>$10^{(n-1)}$, 自由度>1.5；
- 词频>30, 凝固度>$20^{(n-1)}$也能发现很多低频的词汇。

## 中文编码
开发期间遇到一些 python 中文编码处理的坑，记录一下解决方法:

- py 文件开头声明 # encoding:utf-8
- 内部常量 u'' 定义为 unicode
- 所有外部输入 decode('utf-8') 为 unicode 再处理
- 所有外部输出 encode('utf-8')


## 参考资料
https://www.imgless.com/article/70.html
https://www.csdn.net/article/2013-05-08/2815186
http://www.matrix67.com/blog/archives/5044

