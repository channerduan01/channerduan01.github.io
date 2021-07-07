# Causal Inference 因果推断调研

*关键词：Causality/Causation/Casual&Effect 因果，statistics 统计学，epidemiology 流行病，econometrics 计量经济学*

基于两天时间的初步调研，对 Causal Inference 领域简要阐述，并解释一些领域内通用术语和关键概念。另外，针对业务需求，对 DiD (Difference in Differences) 统计方案进行介绍。

## 1. 领域背景
### 1.1 领域起源
主要应用于 流行病学、计量经济学；以下是流行病学、公共卫生领域的两个历史案例：

- 1854年，伦敦爆发霍乱，10天内夺去了500多人的生命。根据当时流行的观点，霍乱是经空气传播的。但是约翰·斯诺（John Snow）医师并不相信这种说法，他认为霍乱是经水传播的。斯诺用标点地图的方法研究了当地水井分布和霍乱患者分布之间的关系，发现在宽街（Broad Street，或译作布劳德大街）的一口水井供水范围内霍乱罹患率明显较高，最终凭此线索找到该次霍乱爆发的原因：一个被污染的水泵。人们把水泵的把手卸掉后不久，霍乱的发病明显下降。约翰·斯诺在这次事件中的工作被认为是流行病学的开端。 

- 1948年-1952年期间，理查·多尔（Richard Doll）和布拉德福·希尔（Bradford Hill）合作进行了一项病例-对照研究（Case-Control Study），通过对癌症患者吸烟史的调查，他们宣布吸烟和肺癌之间有因果联系。其后20多年，他们进行的队列研究（Cohort Study）进一步加强了这一结论。他们的成果为控烟行动提供了科学依据。

### 1.2 典型应用
#### 随机双盲实验 randomized controlled double blind studies
*也常见为 RCT (randomized controlled trials)*

医学上采用该方式验证药物效果。本质上就是控制变量实验，通过完全随机设置两组病人，只对一组给药，观察最终效果差异。这里面的双盲是指：

- 让受试人不知道自己的分组：一组给药、一组给安慰剂，去除“安慰剂”效应带来的影响
- 让实验操作人不知道受试人的分组：排除实验人员期望导致的误差

这类实验通常成本较高，并且存在ethics问题。
另外，目前还没有任何中药通过严格的双盲实验。

### 1.3 领域关联
#### 1.3.1 机器学习 & 因果推断
因果推断是数据驱动决策的基本。

机器学习的很多任务就是在（想）做因果推断，例如线性回归 $y=wX+b$ 也是希望寻找自变量 independent/explanatory variables X 和 dependent/response variable y 之间的联系，在推荐系统、广告系统的应用中，也是为了尽量发现因果关联来提升用户体验、商业营收。同时，因果推断领域也使用机器学习作工具。

但是机器学习更加黑盒（尤其对于NN模型），强调实践效果，而因果推断强调统计和理论；前者在一些较特定化的AI应用领域取得巨大成功，而后者涉及领域更加广泛，应用上更多地和领域知识结合。

[Susan Athey, Machine Learning and Causal Inference, MIT IDSS Distinguished Seminar Series](https://idss.mit.edu/calendar/idss-distinguished-seminar-susan-athey-stanford-university)

[inference](https://www.inference.vc/untitled/)

#### 1.3.2 强化学习 & 因果推断
强化学习的本质是序列决策，核心问题是Credit Assignment（RL模型把获得的reward，自动归因到自己的历史决策序列），这个角度看和因果推断更直接相关。

目前业界也在探索两个交叉领域的结合。

[DeepMind](https://www.jiqizhixin.com/articles/021104)

[华为诺亚ICLR 2020](https://baijiahao.baidu.com/s?id=1654330901725385041&wfr=spider&for=pc)


## 2. Causation 形式化定义
### 2.1 重点在于引入：反现实 counterfactual
对于一种新型药物，我们想观察给一个病人吃药以及不吃药的结果，但我们只能观察到两个结果中的一个（这一问题普遍存在机器学习应用场景中），观察不到的另一个结果称为 反现实 counterfactual。因果推断的理论体系中假设 counterfactual（又称 potential outcomes）存在，在这一前提下进行理论、公式推演。

causal inference 方法在 counterfactual framework（what if）下发展，对具体应用的统计方法再补充假设。例如大部分流行病学分析都假设：no unmeasured confounding（计量经济术语称 no unobserved covariates）。

### 2.2 符号定义 Notation
- 引入二元自变量表示是否给予干预/治疗：a dichotomous treatment variable $A$ (1: treated, 0: untreated) 

- 引入二元因变量表示患者是否死亡：a dichotomous outcome variable $Y$ (1: death, 0: survival)

- 引入反现实结果 counterfactual outcomes（potential outcomes） $Y^{a=1}, Y^{a=0}$ (即不同 $a \in \{0,1\}$ 得到的不同 $Y$)，这些也是随机变量。反现实不是真实观察到的结果，而是假设的规律：例如对于某人，假设他的DNA+所有人生经历，构成了他现在的 $Y^a$，这个反现实结果包含了未来无限可能，但是他最后只能活出（观察到）一个结果。

- 引入真实观察结果 $Y$；并假设 **一致性 Consistency**： $if A_i=a, then Y_i^a=Y^{A_i}=Y_i$，即对于某一用户，采取某一行为后，真实观察到的结果，与对应的反现实结果一致

### 2.3 定义因果
#### 2.3.1 个体因果 Individual causal effect
对某人 $i$，如果有： 
$$Y^{a=1} \neq Y^{a=0}$$ 
则 treatment $A$ 对 outcome $Y$ 构成因果。当然，反现实结果 $Y^{a=0}, Y^{a=1}$ 只存在假设中，现实中只能观测到一个，所以个体因果效应不可得。

#### 2.3.2 平均因果 Average causal effects
在一个确定的人群中，如果整体统计有： 
$$Pr[Y^{a=1} = 1] \neq Pr[Y^{a=0} = 1]$$ 
则 treatment $A$ 对 outcome $Y$ 构成因果。两者相等则表示无关，记为 $Y \perp A$。

（更通用地可写为 $E[Y^{a=1} = 1] \neq E[Y^{a=0} = 1]$）

平均因果按照这个直接定义也是不可得的，但是基于人群级别统计，可以在一定条件下获得（例如做随机分组实验）。

通常从以下3个维度度量因果效应（以下是 $A, Y$ 无关时的取值）：

- 风险差：$Pr[Y^{a=1}=1]-Pr[Y^{a=0}=1]=0$
- 风险比：$\frac{Pr[Y^{a=1}=1]}{Pr[Y^{a=0}=1]}=1$
- 比值比：$\frac{Pr[Y^{a=1}=1]/Pr[Y^{a=1}=0]}{Pr[Y^{a=0}=1]/Pr[Y^{a=0}=0]}=1$

另外，这里补充一般假设：

- 引入 无干涉 No Interfence 假设：即所有人群中所有人都是独立的，$Y_i^{a}$ 不会相互影响、干涉
- 引入 SUTVA (stable-unit-treatment-value) 假设：即假定所有的 treament $A$ 执行一致（例如所有治疗手术都是同一个医生以同样状态进行）

#### 2.3.3 统计误差
平均因果不成立的时候，我们称为 零假设成立：null hypothesis of no average causal effect is true。

这里的因果统计计算时会有两部分 random error：

- Sampling variability：由于采样有限数据集合统计，所以肯定有采样（统计推断）造成的误差。
- Nondeterministic/Stochastic counterfactual：假设所有结果在发生前都是不确定的（参考量子力学的测不准现象，例如某人接受治疗后是否存活取决于一个Bernoulli Distribution），但是观察到的结果是确定的，这会造成误差。

#### 2.3.4 因果 Causation vs. 相关 Association/Correlation
相关性可以直接通过数据统计得到，如果整体统计有： 
$$Pr[Y = 1|A = 1] \neq Pr[Y = 1|A = 0]$$ 
则 treatment $A$ 和 outcome $Y$ 相关 dependent/associated，反之则无关 independent；刻意注意符号的使用，相关性的定义和 反现实 counterfactual 无关，仅仅是根据实际观察得到的条件概率计算。

相关性不等于因果性。主要是 条件概率 $Pr[Y = 1|A = 1]$ 和 边际概率 $Pr[Y^a=1]$ 得差异，如下图（取自 Miguel hernan and James robins, 2020, causal inference）。

![Causation vs. Association](https://img-blog.csdnimg.cn/20200220163901953.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NkZDJ4ZA==,size_16,color_FFFFFF,t_70)


### 2.4 识别因果
#### 2.4.1 随机实验 Random Controlled trials
反现实 $Y^a$ 在真实世界中不存在（我们只能观察到一个结果），所以无法根据 $Pr[Y^{a=1} = 1] \neq Pr[Y^{a=0} = 1]$ 识别因果关系。最简单直接的方法就是进行随机实验（随机采取 treatment $A$，即控制变量AB实验），观察结果差异。

例如：对于一群病人，我们随机地（如抛硬币，以0.5概率）将用户分组，只对一组用户进行治疗，另一组不治疗（或者给予安慰剂），观察结果的相关性（条件概率）：$Pr[Y = 1|A = 1]$ 是否等于 $Pr[Y = 1|A = 0]$；理想随机实验中，相关性即因果性。

**可交换性 Exchangeability** 是这里因果推断的本质依据。理想随机实验中，我们随机切分的两组人群，哪一组接受治疗都能得到一样的结果，不同分组是可交换的；称为：干预动作 treatment $A$ 与 反事实结果 $Y^a$ 无关，记为：$Y^a \perp A\ for\ all\ a$；也称为外生性 exogeneity。具体定义公式如下：
$$Pr[Y^a = 1|A = 0] = Pr[Y^a = 1|A = 1] = Pr[Y^a = 1]\\
Pr[Y^a = 0|A = 0] = Pr[Y^a = 0|A = 1] = Pr[Y^a = 0]$$

结合之前定义的 **一致性** 则有：
$$Pr[Y = 1|A = 1] = Pr[Y^a = 1|A = 1]= Pr[Y^a = 1]\\Pr[Y = 0|A = 0] = Pr[Y^a = 0|A = 0] = Pr[Y^a = 0]$$

即这里的相关性推断等价于因果推断。我们通过随机让 $Y^a$ 与 $A$ 不相关，希望统计到 $Y$ 与 $A$ 相关，进而识别到因果关系。

#### 2.4.2 其他典型实验方法
理想随机实验往往成本很高，甚至完全无法实施（干预动作 treatment 不可控）；我们在引入一定假设的情况下，可以采用其他实验方法。

**交叉实验 Crossover Experiment** 让个体在不同时间段上接受不同 treatment $A$ 并观察结果，可以识别 Individual causal effect；这里需要引入3个强假设：

- 干预/治疗没有持续性影响 no carryover effect of treatment（一般都满足不了）
- 个体的观察结果 $Y_i$ 与时间无关
- 个体的反现实结果 $Y_i^{a}$ 与时间无关

另外，对于 Crossover Randomized Experiment（个体在不同阶段随机接受不同 treatment $A$），能在第一条假设成立的情况下识别 Average causal effects。

**条件随机实验 Conditionally Randomized Experiment** 
引入预测因子 variable $L$ 对原实验人群进行分层 strate，每个分层上再进行随机实验。本质是上认为 $L$ 作为混合因素 confounding factor 同时影响了自变量 $A$、因变量 $Y$，需要消除这一因素。

依据条件可交换性 conditional exchangeability： $$Pr[Y^a = 1|A = 0, L = 1] = Pr[Y^a = 1|A = 1, L = 1]\ ...\ or\ Y^a \perp A|L\ for\ all\ a$$ 可以获得每个分层上的相关性、因果性结论（以及分层 stratum-specific 风险差、风险比、比值比等因果效应测量）；如果分层结果不同，则称为：effect motification by $L$。也可以再通过例如 standardization、ip(inverse probability) weighting 的统计方法把分层统计整合为整体人群统计。

#### 2.4.3 观察性研究 Observational Study
由于随机控制变量实验大多数情况下不现实，实践中我们更多会以观察方法来研究。人类的大部分知识来源于此。其具体操作方法很多，这里使用类比条件随机概率的例子。

在缺失随机控制 randomized treatment assignment 的情况下，引入协变量 covariates $L$，当满足以下3个条件时，Observational Study 可看作是 Conditionallly Randomized Experiment：

- 一致性 Consistency：即观察结果和对应的反现实结果一致
- 可交换性 Excangeability：干预/治疗行为 treatmemt assigment 和观察者、研究者无关，完全取决于协变量 $L$（可能是一组变量；这个条件其实无法证明，只能通过领域知识尽量保证）
- 正性 Positive：不同 协变量 $L$ 条件下的不同 treatment 分组概率都必须大于 0

例如：观察某种药物 $A$ 对病人死亡率 $Y$ 影响时，医生是否给病人用药物取决于病人疾病的严重程度 $L$；对于较严重病人 $L = 1$ 以 0.75 概率随机给药，对于普通病人 $L = 0$ 以 $0.1$ 概率随机给药。本例中，是否用药本身就是有偏的，仅以此统计可能会得出：用药导致病人死亡率大幅升高错误推断（历史上曾有医生因阿司匹林用药后死亡率高，拒绝使用）。

#### 2.4.4 Causal Inference 常规统计方法
都需要结合具体建模场景，满足一定的假设要求和 expert knowledge。

- epidemiology 流行病学：g computation, inverse probability weighting, propensity score matching
- econometrics 计量经济学：instrumental variable, difference in differences, regression discontinuity
- 其他：targeted maximum likelihood learning, super learner

## 3. 双重差分法 Difference in Differences
作为一个 观察性研究 Observational Study 方法，双重差分法使用 面板数据 panel/longitudinal data，测量不同时间段不同分组的结果差异，在一定假设条件下得到因果推断。常应用于计量经济学。

具体来说，引入两个分组：干预组 treatment group $P$ & controlled group 对照组 $S$，然后多次测量（至少干预前测一次，干预后测一次）得到：

- 对照组 事前 结果 $S1$
- 对照组 事后 结果 $S2$
- 干预组 事前 结果 $P1$
- 干预组 事后 结果 $P2$；这是唯一受干预影响的数据，引入 反现实结果 counterfacul outcome，假设无干预情况下会观察到 $Q$

再引入 **平行趋势假设 parallel trend assumption**：对照组、干预组的天然差异（conterfactual）是不变的，事前、事后都一致，即 $P1-S1=Q-S2$。

此情况下，干预带来的结果变化记为：$P2-Q=(P2-S2)-(P1-S1)$；这称为双重差分或两期面板模型，示意图如下（取自 wiki）：

![Differece in Differences](https://img-blog.csdnimg.cn/20200220163924770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NkZDJ4ZA==,size_4,color_FFFFFF,t_70)

更形式化的定义为：
$$Y_{it}=\beta_0+\beta_1I_{it}+\beta_2T_{it}+\beta_3I_{it}T_{it}+\epsilon_{it}$$
其中 $Y_{it}$ 为不同分组 $i$、不同时间 $t$ 观察到的效果，$I$ 为分组虚拟变量（干预组=1，对照组=0），$T$ 为时间虚拟变量（事后=1，事前=0），则有：
$$DiD = D1-D2=(\beta_2+\beta_3)-\beta_2=\beta_3$$

[参考wiki](https://en.wikipedia.org/wiki/Difference_in_differences)


## 4. 参考文献 References

- Miguel hernan and James robins (2020), causal inference, Harvard, 优质入门书籍

- Paul holland (1986) , statisics and causal inference, 领域奠基之作

- Guido imbens and Don rubin, causal inference for statistics & social and biomedical sciences: an introduction

- judea pearl, causality