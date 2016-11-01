#基于内容的推荐系统 Content Based
这一方法基于的产品本身的内容进行挖掘、特征提取，可以通过用户对于特定内容的倾向来推荐给他类似的内容

#协同过滤 Collaborative Filter
这一方法基于抽象级别提取特征，仅仅通过用户的使用记录（包括各种形式的反馈）来提取抽象特征并给出预测。其实这就是 矩阵分解（Matrix Factorization）或者叫 低秩矩阵逼近（Low Rank Matrix Approximation），来求取压缩后的特征矩阵 W（Basis Matrix）和在新的特征空间上的表征 H（Coefficient Matrix）。W、H 是由数据不完全的原始矩阵分解得到，而可以由他们重建原矩阵完成空缺值得预测，而给用户推荐合适的产品

##优点
###能够处理复杂的对象
此方法没有直接分析产品的内在特征，而是通过用户 collaborative 的反馈，来抽取特征；否则，图片、音乐、视频等内容的特征非常难提取的
###新异兴趣的发现
Content Based 的话，将只会推荐用户浏览历史上类似的东西；然后 Collaborative Filter 的话，可能通过和用户类似的其他用户的行为，推荐出一些用户完全没有体验过的东西；本质上是由于 Content Based 的用户经验之间是割裂的，而 Collaborative Filter 的用户经验是大家共享的

##缺点
###冷启动
对于新出现的用户或者产品，无法给出推荐；这是此方法最为严重的问题
###稀疏性
用户操作、反馈往往远远小于产品的是数量，推荐系统的经验矩阵往往是非常稀疏的，过度稀疏的矩阵
###难以做出快速响应
用户有了新的行为之后，无法快速更新模型