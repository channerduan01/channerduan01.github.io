# GBDT 详解
Gradient Boost Decision Tree 是当前非常流行的机器学习算法（监督学习），本文将从渊源起逐层讲解 GBDT，并介绍目前流行的 XgBoost。另有“Adaboost 详解”、“GLM(广义线性模型) 与 LR(逻辑回归) 详解”为本文之基础。

## 0. Hello World
这里列举一个最简单常见的 GBDT 算法。
回归问题，是通过函数 $F(x)$ 拟合目标 $y$；对于 GBDT，每一轮迭代 $t$ 会寻找一个子函数函数 $arg \min_{f_t(x)}(f_t(x)-(F_{t-1}(x)-y))^2$ 集成到 $F(x)$；子函数拟合的目标是模型目前的 residual $(F_{t-1}(x)-y)$，所以宏观上 GBDT 可以会关注到之前迭代中没有处理好的样本，来进一步优化细节，达到越来越好的拟合。（当然也会有 overfitting 的问题了~）

GBDT 中其实包含了非常广泛的思想和应用，本文将详细阐述。

## 1. Some Foundations
GBDT 包含了多种机器学习常见的概念方法，这里将分别介绍几个重要的基础概念。
### 1.1 Gradient Descent
梯度下降是机器学习的 wellknown fashion，对于一个优化目标（loss function），我们要从当前位置（模型状态）走一小步，让 loss 最快的下降（减少）。下面是对 loss function 的一阶泰勒展开：
$$\min_{||v||=1}E(w_t+\eta v)\approx E(w_t)+\eta v\nabla E(w_t)$$

- $w_t$ 表示第 t 轮迭代时，当前的 weight vector
- $v$ 表示走的方向（向量，有 length 约束）
- $\eta$ 表示走的步长（标量，一般是一个较小的正数）

目标其实是找到一个让 loss function 下降最快的方向 $v$，通过其他数学工具可以证明最优的方向是负梯度方向 $v=-\nabla E(w_t)$；$\eta$ 一般是一个超参数，让我们在这个当前最优的方向上走一小步。

### 1.2 Newton Method
牛顿法实际上就是在 Gradient Descent 方法上更进一步，通过 loss function 的二阶导数信息来获取最优的增量 $\Delta w$。
首先对 loss function 做二阶泰勒展开：
$$E(w_t+\Delta w)=E(w_{t+1})\approx g(w_{t+1})= E(w_t)+E'(w_t)(w_{t+1}-w_t)+\frac{1}{2}E''(w_t)(w_{t+1}-w_t)^2$$
接下来是一个 trick，这里对 $g(w_{t+1})$ 做最小化来近似原函数的最小化，其中原函数的函数值及一、二阶导数都被当做常数：$$\frac{d\ g(w_{t+1})}{d\ w_{t+1}}=E'(w_t)+E''(w_t)(w_{t+1}-w_t)=0\\
w_{t+1}=w_t-\frac{E'(w_t)}{E''(w_t)}$$如上直接求得最优的增量 $\Delta w=-E''(w_t)^{-1}E'(w_t)$，其中一阶导数常使用 $g$ 表示梯度（多维度时为向量)，二阶导数常使用 $H$ 表示 Hessian 海森矩阵（多维度时为矩阵），所以标准的更新公式如下：
$$w_{t+1}=w_t-H_t^{-1}g_t=w_t+d_t$$其中 $d_t=-H_t^{-1}g_t$ 被称为 **牛顿方向**，通常直接按上式更新即可，也可以再引入一个步长系数乘以牛顿方向。

牛顿法通常收敛速度比梯度下降方法快，但是需要计算二阶导数，且要求海森矩阵正定，所以后续衍生出了一系列近似的拟牛顿法。

### 1.3 Boosting
提升方法属于 ensemble 集成方法的一类。其在迭代的每一轮中，选择最合适的 weak learner 集成到整体模型中，让整体模型越来越强，提升为 strong learner。注意，它只能串行地训练，因为每一轮迭代都要根据当时模型状况选择最好的子模型。与 Boosting 不同的 Bagging（例如 Random Forest）则可以并发训练。

每一个 weak learner 其实就是一个函数，相对抽象地说，boosting 通过 optimization in function space，找到一组合适的函数集成起来。这个套路类似统计学习中的 Additive Model。

### 1.4 Generalized Linear Model
线性模型 LM 通过 least square $(y-w^Tx)^2$ 完成了 linear predictor（$w^Tx$） 到 expected response $\mathbf{E}[y]$ 的拟合，其中隐含了因变量 $y$ 服从高斯分布。

但是 GLM 把高斯分布推广到了任意 Exponential Family Distribution，例如 Bernoulli Distribution, Laplace Distribution, Gamma Distribution 等等；这极大拓展了 LM，能够处理例如 Classification 的问题。 本质上，这里使用了连接函数把 linear predictor 和 expected response 关联到了一起，所以模型能够学习一个合适的 linear predictor 来预测服从某个分布的 expected response。

GBDT 使用了类似的方式。常规 GBDT 的子模型是 Regression Tree，所以所有子模型的结果加和 $\sum_{t=1}^{T}\alpha_th_t(x)$ 类似于 GLM 中的 linear predictor。之后的套路一致，通过相应的连接函数（往往直接吸收到 loss 中去了），GBDT 也可以学习合适的 linear predictor（即一些列 Regression Tree）来预测服从某个分布的 expected response。当然，GBDT 一般来说比 GLM 拟合效果好很多，因为 GBDT 是在函数空间上寻找一个非线性的最优解，而 GLM 只是在原始的特征空间上寻找一个线性的最优解。
这一套路类似统计学习中 Additive Model to Generalized Additive Model。

### 1.5 Decision Tree
GBDT 基于 GLM，一般使用 Regression Tree；不展开。

## 2. Gradient Boosting Machine
*From AdaBoost to GBDT*

### 2.1 AdaBoost
Gradient Boosting 这一思想来源自 AdaBoost，对于 AdaBoost 来说，Gradient Descent 如下：
$$\min_{\eta} \min_{h_t}\frac{1}{N}\sum_{i=1}^Nexp\{-y_i(\sum_{\tau=1}^{t-1}\alpha_{\tau}h_{\tau}(x_i)+\eta h_t(x_i))\}$$

- $\sum_{\tau=1}^{t-1}\alpha_{\tau}h_{\tau}(x_i)$ 表示已经模型已经集成的部分
- $h_t(x_i)$ 表示需要寻找的 loss 最速下降方向，这里是 Classifier
- $\eta$ 表示需要寻找的，在最速下降方向上最合适的步长

其每次迭代都要选择一个最好的方向，以及方向上最合适的步长（Steepest Descent），最后把一系列的方向（$h_t(x)$）和步长（写作 $\alpha_t$）集成在一块。优化过程本质上是在 Exponential Error 的 loss 函数上做梯度下降，这也是 Gradient Boosting，所以常规的 AdaBoost 其实也可称作 GBM 或 GBDT。

### 2.2 Gradient Boosting
Gradient Boosting 是通过 GLM 思路对 AdaBoost 推广，Gradient Descent 如下：
$$\min_{\eta} \min_{h_t}\frac{1}{N}\sum_{i=1}^NLoss(\sum_{\tau=1}^{t-1}\alpha_{\tau}h_{\tau}(x_i)+\eta h_t(x_i),\ y_i)$$

这里结合了 GLM，和之前 AdaBoost 有俩个区别：

- $h_t(x_i)$ 使用 Regressor，这里 $\sum_{t=1}^{T}\alpha_th_t(x)$ 类似 GLM 中的 linear predictor
- loss function 没有限制，这里表达为 $Loss(\sum_{t=1}^{T}\alpha_th_t(x_i),y_i)$；通常基于 GLM 的套路根据 response variable $y$ 的分布，结合对应的 link function 来确认 loss

GBM 的应用场景被大幅拓展了，不局限与 exponential error 或者 least square；正如 GLM 一样拓展到了各种 $y$ 分布上。一句话：Regressor here is not just for regression, it all depends on how you define the objective function!

### 2.3 Subtle Different of Gradient Descent
AdaBoost 和 GBDT 在梯度下降的细节上有不一致。本质原因在于函数方向 scale，注意，我们寻找的 $h(x)$ 只是一个方向，scale 上没有意义。

AdaBoost 使用 Classifier 做子模型 $h(x)$，这天然可以在直接函数空间中表示一个方向，不需要约束 scale。所以 AdaBoost 可以固定 $\eta$ 以 loss function  $\displaystyle\min_{h_t}\frac{1}{N}\sum_{i=1}^Nexp\{-y_i(\sum_{\tau=1}^{t-1}\alpha_{\tau}h_{\tau}(x_i)+\eta h_t(x_i))\}$ 直接解出最速下降方向 $h_t(x)$，这是很直接、丝滑的套路。

GBDT 使用 Regressor 做子模型，这必须加以约束才能在函数空间表示一个方向，正如梯度下降标准套路 $\min_{||v||=1}E(w_t+\eta v)\approx E(w_t)+\eta v\nabla E(w_t)$ 中的 $||v||=1$。如果不加以 scale 约束，这个方向没有意义，并且没法解出来的（可能包含一堆无穷大和无穷小）。所以 GBDT 间接地解决了这个问题：以当前模型的负梯度方向为标杆，寻找一个尽量平行于负梯度方向的 $h_t(x)$。实现上是引入一个 least square $\displaystyle\min_{h_t}(h_t(x)-(-\nabla E))^2$，解出一个接近最优方向的 $h_t(x)$。其实这个套路还能找到一些其他的解释，总的来说，就是把带约束的优化问题转为不带约束的优化问题，让我们能较容易地解出作为方向的 regressor $h_t(x)$。

### 2.4 Standard Gradient Boosting for GBM
*GBDT 原文论文 Gradient Boost 套路如下*：
$$\begin{aligned}
&F_0(x)=arg \min_p\sum_{i=1}^NL(y_i,p)\\
&For\ t=1\ to\ T\ do:\\
&\ \ \ \ \tilde{y_i}=-[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}]_{F(x)=F_{t-1}(x)},\ i=1,N\\
&\ \ \ \ a_t=arg\ \min_{\alpha,\ \beta}\sum_{i=1}^N[\tilde{y_i}-\beta h(x_i;\alpha)]^2\\
&\ \ \ \ p_t=arg\ \min_p\sum_{i=1}^NL(y_i,F_{t-1}(x_i)+ph(x_i;a_t))\\
&\ \ \ \ F_t(x)=F_{t-1}(x)+p_th(x_i;a_t))\\
&endFor
\end{aligned}$$
先对模型进行一个初始化，这相当于确认一个先验，例如普通回归问题的话就使用样本的 $y$ 均值，而二分类使用样本为 positive 的统计概率。

之后有 $T$ 轮迭代，每次迭代步骤如下：

- 计算当前模型的负梯度方向 $\tilde{y_{\ }}$
- 寻找一个尽量平行于此方向的子模型 $h(x)$，是要找一个方向，scale 不重要；$h(x)$ 我们通常假设是某种模型，例如回归树，所以这里寻找的是 $h(x, a_m)$ 的模型参数 $a_m$
- 确定了模型（方向）后，在原始 loss 上求解出最合适的步长，即 Steepest Descent 的 greedy 套路；是为了尽快下降，减少我们需要集成的子模型数量

## 4. Gradient Boosting Decision Tree
*there is an interesting trick with tree*

GBDT 可以说是对 GBM 的一种实现，其子模型使用 Regression Tree。但是，基于 Decision Tree 的特点，Friedman 原论文对其最优化过程做出了非常有意思的调整（**$\eta$ 的求解步骤**），拟合能力大幅增强。

### 4.1 Boosting Tree
之前也有提到，寻找函数方向 $h(x)$，实际上是在寻找一组合适模型参数。当使用 Regression Tree 作 $h(x)$，可以把模型参数看做 $h(x;\{b_j,R_j\}^J_1)$：

- 树模型把样本划分到 J 个 region 中（即 J terminal nodes）
- 每个 region 使用 $R_j$ 表示（实际上暗含了一组规则来切分出这个 region）
- 每个 region 对应一个输出 $b_j$

分割出的 $J$ 是 disjoint 的，所以有：
$$h(x;\{b_j,R_j\}^J_1)=\sum_{j=1}^Jb_j1(x\in R_j)$$

$1(x\in R_j)$ 表示样本属于这一 region，这里可以把一个函数 $h(x)$ 等价看作 $J$ 个函数。

### 4.2 More of $\eta$
标准 GBM 套路中，求解最适步长 $\eta$ 只是一个实数，目标如下：
$$\min_{\eta} \sum_{i=1}^NLoss(F_{t-1}(x_i)+\eta h_t(x_i),\ y_i)$$

但是，对于 GBDT，每一个 $h(x)$ 包含 $J$ 个 region，可看作 $J$ 个函数；我们可以对每一个 region 求解一个合适的步长！目标如下：
$$\{\gamma_{jm}\}_1^J=arg \min_{\{\gamma_{j}\}_1^J} \sum_{i=1}^NLoss(F_{t-1}(x_i)+\sum_{j=1}^J\gamma_j1(x\in R_j)  ,\ y_i)\\
\gamma_{jm}=arg \min_{\gamma}\sum_{x_i\in R_{jm}}Loss(F_{t-1}(x_i)+\gamma  ,\ y_i)$$

其中 $\gamma_j$ 实际就是 $\eta_jb_j$，这里把这俩揉一块了；本质上我们使用 $J$ 个步长 $\eta_j$ 来更好地优化 loss，拟合目标。
并且，这 $J$ 个 region 是 disjoint 的，所以各 region 可以分开求一个单变量 $\gamma_j$ 的优化问题。

### 4.3 Example of Binary Classification
这里以二分类问题常用的 loss（negative binomial log-likelihood）实例分析 GBDT 优化过程（又被称为 LogitBoost）。
Loss 函数即  $L(y,F)=log(1+\exp{(-2yF)}),\ y\in\{-1,1\}$

*GBDT 原文论文 二分类训练套路如下*：
$$\begin{aligned}
&F_0(x)=\frac{1}{2}log\frac{1+avg(y)}{1-avg(y)}\\
&For\ t=1\ to\ T\ do:\\
&\ \ \ \ \tilde{y_i}=2y_i/(1+\exp{2y_iF_{t-1}(x_i)}),\ i=1,N\\
&\ \ \ \ \{R_{jt}\}^J_1=get\ tree\ structure\ with\ J\ terminal\ nodes\ (\{\tilde{y_i},x_i\}^N_1)\\
&\ \ \ \ \gamma_{jt}=\frac{\sum_{x_i\in R_{jt}}\tilde{y_i}}{\sum_{x_i\in R_{jt}}\tilde{|y_i|}(2-\tilde{|y_i|})}\\
&\ \ \ \ F_t(x)=F_{t-1}(x)+\sum_{j=1}^J\gamma_{jt}1(x\in R_{jt})\\
&endFor
\end{aligned}$$

- 其中负梯度 
$$\begin{aligned}
\tilde{y_i}\ &=-[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}]_{F(x)=F_{t-1}(x)}\\
&=-\frac{-2y_i\exp\{-2y_iF_{t-1}(x_i)\}}{1+\exp\{-2y_iF_{t-1}(x_i)\}}\\
&=\frac{2y_i}{1+\exp\{2y_iF_{t-1}(x_i)\}}
\end{aligned}$$

- 其中 terminal node 最优值的计算公式 $$\gamma_{jt}=\frac{\sum_{x_i\in R_{jt}}\tilde{y_i}}{\sum_{x_i\in R_{jm}}\tilde{|y_i|}(2-\tilde{|y_i|})}$$ 是对步长的优化目标
$$\gamma_{jt}=arg \min_{\gamma}\sum_{x_i\in R_{jt}}log(1+\exp\{-2y_i(F_{t-1}(x_i)+\gamma)\})$$ 利用 Newton-Raphson step 近似推导出的结果。

另外，这里额外提一下 K 分类的问题。K 分类需要训练 K 组不同的树（即每轮迭代需要训练 K 颗不相关的树）；每一组树结果叠加成 linear predictor 后输出一个判别概率，这和广义线性模型中处理 K 分类的思路是一模一样的。

## 5. XgBoost
### 5.1 全新的 loss function（被称为 objective function）
XgBoost 是对陈天奇对 GBDT 的一个非常高效的实现；不仅如此，其最重要的一个新特性是对 Regularization 的强化：**heuristic to objective**。传统的树模型生成结构时，往往只考虑 impurity，把 complexity 的控制完全交给 heuristic 策略；但是 XgBoost 将部分控制吸收到了优化目标 objective（即 loss function）。

具体实现上涉及两个细节：

- 使用了牛顿法，通过一阶、二阶导数优化 loss function
- 向 loss function 中加入了 regularizaiton term

在传统 GBM 的优化思路中，我们是每次迭代中依次找到合适的函数方向 $f_t$ 以及最佳的步长 $\eta$，以梯度下降优化 loss function。
$$\min_{\eta} \min_{h_t}\frac{1}{N}\sum_{i=1}^NLoss(F_{t-1}(x_i)+\eta f_t(x_i),\ y_i)$$
XgBoost 也是 GBM，但是使用牛顿法（后续详解），直接寻找一个函数 $f_t$ 作牛顿方向来优化 loss function，并且，在优化目标中直接加入 regularization term $\Omega$ 来控制模型复杂度（heuristic to objective 思想）；所以最终的 loss function 在这里变成 objective function。
$$\begin{aligned}
Obj^{(t)}&=Loss+\Omega\\
Obj^{(t)}&=\sum_{i=1}^NLoss(F_{t-1}(x_i)+f_t(x_i),y_i)+\sum_{\tau=1}^t\Omega(f_{\tau})\\
Obj^{(t)}&=\sum_{i=1}^NLoss(F_{t-1}(x_i)+f_t(x_i),y_i)+\Omega(f_t)+constant
\end{aligned}$$

XgBoost 和之前 GBM 的一个区别是这里要求上述的 Loss 函数二阶可导。接下来就是对这个 objective function 的各种变化，确定最优化求解的步骤。

### 5.2 Newton Boosting Tree
#### 5.2.1 **Taylor Expansion Approximaiton of Loss**
1.2 中有详细解释 Newton Method。首先，Recall 函数的二阶泰勒展开公式为：
$$f(x_0+\Delta x)\approx f(x_0)+f'(x_0)\Delta x+\frac{1}{2}f''(x_0)\Delta x^2$$
所以对 objective function 近似如下：
$$Obj^{(t)}\approx \sum_{i=1}^N[Loss(F_{t-1}(x_i),y_i)+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)+constant\\
g_i=\frac{d\ Loss}{d\ F_{t-1}}\bigg|_{F_{t-1}(x_i),y_i},\ h_t=\frac{d^2\ Loss}{d\ F_{t-1}}\bigg|_{F_{t-1}(x_i),y_i}$$
上式中的 $g_i$ 和 $h_i$ 分别为 loss 的一阶、二阶导数在 $(F_{t-1}(x_i),y_i)$ 上的取值；注意，是根据导数函数，直接在模型上一轮迭代结果 $F_{t-1}(x_i)$ 和 label $y_i$ 上计算出来的数值。
去除末尾的 constant，并去除对于第 t 轮迭代已经是常数的 $Loss(F_{t-1}(x_i),y_i)$，上式整理为：
$$Obj^{(t)}\approx\sum_{i=1}^N[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)$$即寻找一个函数 $f_t$ 最小化上述目标。

#### 5.2.2 **Tree Loss**
类似于 4.1 中将目标函数按照 terminal node 切为一系列函数 $h(x;\{b_j,R_j\}^J_1)=\sum_{j=1}^Jb_j1(x\in R_j)$，XgBoost 使用了同样套路，只是定义符号有所不同：
$$f_t(x)=w_{q(x)},\ w\in R^J,\ q:R^d\rightarrow\{1,2,...,J\}$$
即函数 $f_t$ 可以视为 $J$ 个 terminal node，每个 node 对应一个值 $w_j$；并且 Tree Structure 函数 $q(x)$ 可以把样本映射到具体的某个 terminal node。

为了方便目标函数改写，这里定义集合 $I_j=\{i|q(x_i)=j\}$ 表示归属于某一 terminal node $j$ 的所有样本 index；并且再定义 $G_j=\sum_{i\in I_j}g_i$ 和 $H_j=\sum_{i\in I_j}h_i$，表示同一 terminal 上样本的一阶、二阶导数值之和。

现在，对原 objective function 做如下改写：
$$\begin{aligned}
Obj^{(t)}&\approx\sum_{i=1}^N[g_iw_{q(x_i)}+\frac{1}{2}h_iw_{q(x_i)}^2(x_i)]+\Omega(f_t)\\
&=\sum_{j=1}^J[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in I_j}h_i)w_j^2]+\Omega(f_t)\\
&=\sum_{j=1}^J[G_jw_j+\frac{1}{2}H_jw_j^2]+\Omega(f_t)
\end{aligned}$$
这里基于 Tree Structure，把优化目标从 $N$ 个样本的 cost，改写成了 $J$ 个 terminal node 的 cost。

#### 5.2.3 **Tree Regularization**
基于树模型，定义 $f_t$ 的复杂度如下：
$$\Omega(f_t)=\gamma J+\frac{1}{2}\lambda\sum_{j=1}^Jw_j^2$$通过两个超参数 $\gamma$ 和 $\lambda$ 控制约束的程度（balancing），分别控制了树的叶子节点数量，以及叶子节点值的平方和（其实也会控制数量)；总的来说就是让叶子越少，上面的值越小越好。

整合上 regularisation term 后，objective function 为：
$$\begin{aligned}
Obj^{(t)}&=\sum_{j=1}^J[G_jw_j+\frac{1}{2}H_jw_j^2]+\gamma J+\frac{1}{2}\lambda\sum_{j=1}^Jw_j^2\\
&=\sum_{j=1}^J[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2]+\gamma J
\end{aligned}$$上式把部分 regularisation term 直接被收到 loss 中了。这就是 XgBoost 的 objective function 完全体，非常简洁。并且 $g_i$ 和 $h_i$ 蕴含的是任意二阶可导的 loss function，所以也是一个很通用的 GBM。

#### 5.2.4 **Newton Method**
按照牛顿法标准套路，我们可以进一步化简上述 objective function。固定住 tree structure 和超参数，对于某一 terminal node $j$ 有：
$$E=G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2+constant$$通过求导直接取得最优解：
$$\frac{d\ E}{d\ w_j}=G_j+(H_j+\lambda)w_j=0\\
w_j^*=-\frac{G_j}{H_j+\lambda}$$本质上这个叶子节点取值 $w_j^*$ 就是牛顿方向，以一个最合适的增量让原函数 objective function（近似地）取到最小值。
**所以，对于任一确定的 tree structure，我们可以得到各个 terminal node 的最优 $w_j$ 取值。**

接下来，一个有点 tricky 的方法：我们直接把 $w_j^*$ 代入 objective function，将得到一个对于 tree structure 的评估函数：
$$\begin{aligned}
Obj^{(t)}&=\sum_{j=1}^J[G_jw^*_j+\frac{1}{2}(H_j+\lambda){w^*_j}^2]+\gamma J \\
&=\sum_{j=1}^J[-\frac{G_j^2}{H_j+\lambda}+\frac{1}{2}(H_j+\lambda){(-\frac{G_j}{H_j+\lambda})}^2]+\gamma J \\
&=-\frac{1}{2}\sum_{j=1}^J\frac{G_j^2}{H_j+\lambda}+\gamma J
\end{aligned}$$**对于任意一个 tree structure，上述 objective function 将给出这一 tree structure 的最小 cost，所以这是构建树结构的基础评估函数。**

最终的 XgBoost 训练流程，正是通过这一评估函数生成合适的 Tree Structure，然后再求出其各个叶子节点对应值 $w_j^*$。

### 5.3 XgBoost Algorithm
理论基础见 5.2，具体应用中，每轮迭代生成一个另当前 objective 尽量最小的 Regression Tree。GBDT 的原论文感觉更强调各类 loss function 上的应用，展现类似 GLM 的通用性；而 XgBoost 非常关注具体的 Decision Tree 的生成细节。

单个 Tree 也是一个 greedy（都是这么玩的~）的生成过程，对于所有当前叶子节点，会扫描所有 feature 维度所有可能的 split，来找到一个最优 split 使得 Gain 最大（即 loss 减少最多）：
$$Gain=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma$$
上式即 5.2 中推导的评估函数，使用 split 后俩个节点的 loss 和减去之前节点的 loss，并附加上增加一个节点的 regularizer $\gamma$；作为 split 的收益评估。

和各类树模型一致，这里也是一个纯 search 的过程，细节上会涉及连续值、离散值、缺失值的不同应对策略，以及各式 heuristic 策略控制树的复杂度，这里不做展开。

另外，XgBoost 在集成每一轮迭代时建议使用 shrinkage $\epsilon=0.1$，即 $F_t=F_{t-1}+\epsilon f_t$，很保守~

### 5.4 Some regularization of GBDT
这里进一步列举几个常见 regularization 思路，在 XgBoost 以及后续的 LightGBM 上都有体现：

- Objective Function 控制复杂度
可以说是 XgBoost 的主推特性；传统的 decision tree 只通过 heuristic 策略控制

- split 策略控制复杂度，很通用，例如：
 - Maximum of Tree Depth
 - Maximum of Terminal Nodes, 
 - Minimum of Gain of Split,
 - Minimum of Node Size

- Pruning
又分为前剪枝 Pre-Pruning 和后剪枝 Post-Pruning：
 - 前剪枝的话使用一些 split 策略，直接停止一些节点的生长，但是有可能会遇到 greedy 的问题，有些路径可能一开始 gain 很小但是后续逆袭；
 - 后剪枝的话先完全生成（限制 depth）树结构，再逐渐递归地从叶子节点剪枝（也会有一系列策略、算法），这个结构太大，较容易 overfitting。

- 行采样，row-subsampling
正如 Random Forest 这类 bootstrap 套路；抗 overfitting

- 列采样，column-subsampling
这也是很多 ensemble 方法的通用套路，包括 Isolation Forest；抗 overfitting~

- Shrinkage
每次学习到的子模型都以一定比例衰减权值，刻意降低拟合的效率来避免过拟合；GBDT 通用套路

- Early Stop
机器学习通用套路，对于 GBDT 这种 emsemble 模型更方便；由于模型可加性，训练完成后，最终模型到底集成多少棵树，是可以随时调整的~


## Reference
XgBoost introduction: https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf

Origin of GBDT: Greedy function approximation a gradient boosting machine. J.H. Friedman


