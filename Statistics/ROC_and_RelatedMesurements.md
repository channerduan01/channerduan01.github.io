#ROC (Receiver Operating Characteristic Curve) 受试者特性曲线
##基本度量
整个度量评估针对的是 二元分类 Binary Classification 的问题，包含 Positive 和 Negative。分类器建立了俩个类型的分布，但是在实用case中，他们是会有重叠 overlap 的，overlap 的面积取决于问题本身分类的难度以及分类器的能力。最终的分类由一个阈值决定，我们调整这个阈值可以产生不同的分类效果，这是一个权衡、平衡的过程，可由 ROC 来进行描述。想象在二维空间里，俩个有重叠的分布被一条直线切割成左右俩部分，左侧 positive，右侧 negative，一些重要指标如下：  

**Positive**，真正 positive 的数量  
**Negative**，真正 negative 的数量  
**True Positive**，左侧 positive 的数量  
**False Positive (type 1 error)**，左侧 negative 的数量  
**True Negtive**，右侧 negative 的数量
**False Negtive (type 2 error)**，右侧 positive 的数量  

**True Positive Rate (ROC 的竖轴) = Recall 召回率（查全率） = Sensitivity**，$\displaystyle\frac{TP}{P}$  
**False Positive Rate (ROC 的横轴) = Fall-out**，$\displaystyle\frac{FP}{N}$
**Precision 查准率**，$\displaystyle\frac{TP}{TP+FP}$  
**ACC (Accuracy)**，$\displaystyle\frac{TP+TN}{P+N}$  
**F1**，it is the average of **recall** and **precision**, which is a single metrics to define the performance of a classifier.  
$\displaystyle\frac{2TP}{2TP+FN+FP}$

###AUC (Area Under Curve)
这个是用于衡量 binary classifier 性能的指标，越大越好，[0.5, 1]，0.5 是随机分类的结果，不能比 0.5 还差（还差的情况下对判别标准取反就好，之后得到的 AUC 应该是 1-原来错误的AUC）

注意！AUC 对不平衡数据不敏感，这是一大优势。

#PRC (Precision-Recall Curve)
当我们仅对 Binary Classification 中的某一类比较感兴趣的时候，PRC 是一个比较重要的考量。一下是两个重要差异：
- 1. ROC 衡量的是分类器正确划分两个类别的能力，而 PRC 强调分类器正确、精确识别出某一类别的能力。
- 2. ROC 衡量综合能力，对 imbalance 不敏感，而 PRC 对 imbalance 非常敏感

PRC 可能会表现的非常不稳定，最好画出曲线观察一下。

## 应用举例
例如 反作弊 问题中，我们非常关注 作弊positive 的 precision，因为我们对 False-Alarm 非常敏感，误识别会严重伤害相关业务。ROC 衡量的是综合分类能力且对imbalance不敏感，当数据集中绝大部分数据是negative时，对 Precision 构成严重影响的大量 False-Alarm 可能只对应了非常小的 Fall-out 而无法被 ROC 感知。这种场景下，Precision 和 Recall 的平衡考量成为分类器的关键。所以我们考虑用 PRC 来评估分类器的性能。

###AP (Average Precision) = MAP (Mean Average Precision)
等同于 ROC 的 AUC 指标，标识 PRC 曲线下方的面积，用单一值反应出分类器查全查准的能力，取值 [0, 1]，越大越好











