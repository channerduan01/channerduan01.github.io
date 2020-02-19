# Causal Inference 因果推断

关键词：Causality/Causation/Casual&Effect 因果律，statistics 统计学，epidemiology 流行病，econometrics 计量经济学

基于2天时间的初步调研，对 Causal Inference 领域简要阐述，并解释一些领域内通用术语和关键概念。另外，针对业务需求，对 DiD (Difference in Differences) 统计方案进行介绍。

## 1. 领域背景

### 1.1 领域起源

主要应用于 流行病学、计量经济学，以下是流行病学、公共卫生领域的两个历史案例：

- 1854年，伦敦爆发霍乱，10天内夺去了500多人的生命。根据当时流行的观点，霍乱是经空气传播的。但是约翰·斯诺（John Snow)医师并不相信这种说法，他认为霍乱是经水传播的。斯诺用标点地图的方法研究了当地水井分布和霍乱患者分布之间的关系，发现在宽街（Broad Street，或译作布劳德大街)的一口水井供水范围内霍乱罹患率明显较高，最终凭此线索找到该次霍乱爆发的原因：一个被污染的水泵。人们把水泵的把手卸掉后不久，霍乱的发病明显下降。约翰·斯诺在这次事件中的工作被认为是流行病学的开端。 

- 1948年-1952年期间，理查·多尔（Richard Doll）和布拉德福·希尔（Bradford Hill）合作进行了一项病例-对照研究（Case-Control Study），通过对癌症患者吸烟史的调查，他们宣布吸烟和肺癌之间有因果联系。其后20多年，他们进行的队列研究（Cohort Study）进一步加强了这一结论。他们的成果为控烟行动提供了科学依据。





### 1.2 典型应用

### 1.3 学科关联

#### 机器学习 vs. 因果推断



 Susan Athey, Machine Learning and Causal Inference, MIT IDSS Distinguished Seminar Series, <https://idss.mit.edu/calendar/idss-distinguished-seminar-susan-athey-stanford-university/>



#### 强化学习 vs. 因果推断





<https://www.one-tab.com/page/5anvnWtVTP2HkSLPPBzQlg>

<https://www.one-tab.com/page/Jt2W_3ibRBiAGNtrAHsdXA>



## 2. Causation 形式化定义



### 随机双盲实验 randomized controlled double blind studies

（常见 RCT randomized controlled trials）

医学上采用该方式验证药物效果。本质上就是控制变量实验，通过完全随机设置两组病人，只对一组给药，观察最终效果差异。这里面的双盲是指：

- 让受试人不知道自己的分组：一组给药、一组给安慰剂，去除“安慰剂”效应带来的影响
- 让实验操作人不知道受试人的分组：排除实验人员期望导致的误差

目前还没有任何中药通过严格的双盲实验。这类实验通常成本较高，并且存在ethics问题。

### Causal Inference 常规统计方法

都需要结合具体建模场景，满足一定的假设要求和 expert knowledge。

- epidemiology 流行病学：g computation, inverse probability weighting, propensity score matching
- econometrics 计量经济学：instrumental variable, difference in differences, regression discontinuity
- 其他：targeted maximum likelihood learning, super learner



### Causality 面临的根本问题：反现实 counterfactual

对于一种新型药物，我们想观察给一个病人吃药以及不吃药的结果，但我们只能观察到两个结果中的一个，观察不到的另一个结果称为 反现实 counterfactual。

所以现在的 causal inference 方法在 counterfactual framework（又称 potential outcomes）下发展，且具体的统计方法都需要补充假设；例如大部分流行病学分析都假设：no unmeasured confounding（计量经济术语称 no unobserved covariates）。



###  Association vs. Causation

### Causality Definition

#### Notation

- Consider a dichotomous treatment variable $A$ (1: treated, 0: untreated) 
- and a dichotomous outcome variable $Y$(1: death, 0: survival)

- Let $Y^{a=1}$ (read $Y$ under treatment $a$ = 1) be the outcome variable that would
  have been observed under the treatment value $a$ = 1; And also $Y^{a=0}$. $Y^{a=1}$, $Y^{a=0}$ are also random variables.

#### 个体因果 Individual causal effect

*the treatment $A$ has a causal effect on an individual’s outcome $Y$ if
$Y^{a=0} \neq Y^{a=0}$ for the individual*

变量 $Y^{a=0}, Y^{a=0}$ 也称为反现实结果 counterfactual outcomes（或 potential outcomes），因为在用户个体层面，我们只能观察到两个结果中一个 missing data，所以个体因果推断是无法计算的。

我们定义真正观测到的结果为 $Y$，有 $Y=Y^{A}$；该公式又称为 一致性 consistency。

#### 平均因果 Average causal effects

需要3部分信息来定义:

- an outcome of interest $Y$

-  the actions $a$ = 1 and $a$ = 0 to be compared

-  a well-defined population of individuals whose outcomes $Y$=0 and $Y$=1 are to be compared

具体公式定义如下：

$Pr[Y^{a=1} = 1] \neq Pr[Y^{a=0} = 1]$ 或是更通用的（non-dichotomous outcomes） $E[Y^{a=1} = 1] \neq E[Y^{a=0} = 1]$

平均因果不成立的时候，我们称为 零假设成立：null hypothesis of no average causal effect is true。另一方面，这里的因果统计计算时会有2部分 random error：

- Sampling variability：由于采样有限数据集合统计，所以肯定又采样造成的误差
- Nondeterministic counterfactual：使用了个性确定性的结果（0、1事件）统计，理论上不同个体的结果都有一定 随机性 stochastic（可能是一个 Bernoulli Distribution ）



## References

- Miguel hernan and James robins (2020-), causal inference,  Harvard 优质入门书籍

- Paul holland (1986 ) , statisics and causal inference 领域奠基之作

- Guido imbens and Don rubin, causal inference for statistics、social and biomedical sciences: an introduction

- judea pearl, causality