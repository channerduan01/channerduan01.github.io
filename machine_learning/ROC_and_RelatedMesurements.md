#ROC (Receiver Operating Characteristic Curve) 受试者特性曲线

##基本度量
整个度量评估针对的是 二元分类 Binary Classification 的问题，包含 Positive 和 Negative。分类器建立了俩个类型的分布，但是在实用case中，他们是会有重叠 overlap 的，overlap 的面积取决于问题本身分类的难度以及分类器的能力。最终的分类由一个阈值决定，我们调整这个阈值可以产生不同的分类效果，这是一个权衡、平衡的过程，可由 ROC 来进行描述。想象在二维空间里，俩个有重叠的分布被一条直线切割成左右俩部分，左侧 positive，右侧 negative，一些重要指标如下：  

**Positive**，真正 positive 的数量  
**Negative**，真正 negative 的数量  
**True Positive**，左侧 positive 的数量  
**False Positive (type 1 error)**，左侧 negative 的数量  
**True Negtive**，右侧 negative 的数量
**False Negtive (type 2 error)**，右侧 positive 的数量  

**True Positive Rate (ROC 的竖轴) = Recall 召回率 = Sensitivity**，$\displaystyle\frac{TP}{P}$  
**False Positive Rate (ROC 的横轴) = Fall-out**，$\displaystyle\frac{FP}{N}$  
**ACC (Accuracy) 精确度**，$\displaystyle\frac{TP+TN}{P+N}$  
**Precision**，$\displaystyle\frac{TP}{TP+FP}$  
**F1**，it is the average of **recall** and **precision**, which is a single metrics to define the performance of a classifier.  
$\displaystyle\frac{2TP}{2TP+FN+FP}$

###AUC (Area Under Curve)
这个是用于衡量分类器性能的指标，越大越好，[0.5, 1]，0.5 是随机分类的结果，不能比 0.5 还差。










