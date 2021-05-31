# 统计置信理论初探

## 1. 基础概念
首先铺垫、强调一些必要的统计概念。

### 1.1 区分总体与采样
- Population 总体：又称为全体或者整体，是指由多个具有某种共同性质的事物的集合。
- Sample 样本：是指总体中随机抽取的个体。

一般情况下，我们观察不到“总体”，只能观察到“样本”；但是可以根据样本的情况来推断总体的规律。例如我们需要了解全校男生身高水平，可以采样50个男生来统计推断。

额外地，样本“随机抽取”假设服从独立同分布 “i.i.d.”（independent and identically distributed），即

- independent 独立：不同样本抽样没有关系，例如投骰子每一次都是独立事件
- identically 同分布：且服从同一个分布，例如同一个骰子每一次投各面出现概率都一致

### 1.2 均值与方差
假设 $Y$ 是我们关心的 random variable 随机变量，例如身高。

通常无法直接统计总体：

- population mean 总体均值（数学期望）：$\mu = \frac{1}{\infty}\sum_{i=1}^{\infty}y_i$
- population variance 总体方差：$\sigma^2 = \frac{1}{\infty}\sum_{i=1}^{\infty}(y_i-\mu)^2$

只能统计样本（引入样本数量 $n$）：

- sample mean 样本均值：$\widehat\mu = \frac{1}{n}\sum_{i=1}^{n}y_i$
- sample variance （无偏）样本方差：$\widehat\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i-\mu)^2 \approx \frac{1}{n-1}\sum_{i=1}^{n}(y_i-\widehat\mu)^2$（样本方差也常被记作 $s^2$）
- sample standard deviation 样本标准差 即 $\widehat\sigma$

额外地，样本方差评估为啥使用“$\frac{1}{n-1}$”而不是“$\frac{1}{n}$”？前者是无偏样本方差而后者是有偏样本方差。样本均值统计等价于 $\widehat\mu=argmin_{c}\frac{1}{n}\sum_{i=1}^n(y_i-c)^2$，其本质上是在最小化有偏样本方差，会引入偏差造成样本方差被低估。所以需要纠正去偏，详见 [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)


## 2. SEM 标准误差
置信度中最为核心的概念，standard error of the sample mean。

### 2.1 定义
SEM 标识了 sample mean 对于 population mean 的估计误差（误差越小估计越准），是近似估计。

$SEM=\Large\frac{\widehat\sigma}{\sqrt{n}}$


“SEM is standard deviation of its sampling distribution”，其估计的是无穷多 sample mean（不同样本集的估计值可能不同）相对于 population mean（真相只有一个） 的标准差。

例如，我们需要估计全校男生身高平均值，我们随机采样100个男生之后可以计算出相应的 sample mean 即 $\widehat\mu_1$，然后我们重新随机采样100个男生计算出 sample mean $\widehat\mu_2$... 这一系列 sample mean 的标准差即 $SEM=\sqrt{\frac{1}{\infty}\sum_{i=1}^{\infty}(\widehat\mu_i-\mu)^2}$。

### 2.2 注意区分 SD vs. SEM
- SD indicates how accurately the mean represents sample data. 
- SEM includes statistical inference based on the sampling distribution. SEM is the SD of the theoretical distribution of the sample means (the sampling distribution).

### 2.3 理论推导
基于中心极限定理 Central Limit Theorem：

- 样本平均值约等于总体平均值。
- 不管总体是什么分布，任意样本平均值都会围绕在总体平均值周围，并且呈正态分布。

具体来说，样本均值和总体均值的差的 $\sqrt{n}$ 倍（$n$为样本量，要求$n$足够大）近似于一个正态分布 $N(0,\sigma^2)$ 即：

$$\sqrt{n}(\widehat\mu-\mu)\ \thicksim\ N(0,\sigma^2)\\
(\widehat\mu-\mu)\ \thicksim\ N(0,\frac{\sigma^2}{n})\\
\widehat\mu\ \thicksim\ N(\mu,\frac{\sigma^2}{n})\\
SEM=\sqrt{\frac{\sigma^2}{n}}=\frac{\sigma}{\sqrt{n}}\approx\frac{\widehat\sigma}{\sqrt{n}}
$$

详见 [Central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)

### 2.4 代码演示实例
#### 中心极限定理可视化
```
import matplotlib.pyplot as plt

# god view
population_std = 10
population_mean = 100

def run_experiment(n): 
    return np.random.normal(population_mean, population_std, n)

np.random.seed(42)

plt.figure(figsize=(8,5))
freq, bins, img = plt.hist([run_experiment(n=500).mean() for _ in range(10000)], bins=40, label="Sample Means")
plt.vlines(true_mean, ymin=0, ymax=freq.max(), linestyles="dashed", label="Population Mean", color="orange")
plt.legend();
```
#### SEM随样本量变化可视化
```
import matplotlib.pyplot as plt
import math
import numpy as np

# god view
population_std = 10
population_mean = 0

def run_experiment(n): 
    sample_list = np.random.normal(population_mean, population_std, n).tolist()
    sample_mean = sum(sample_list)/n
    sample_standard_deviation = math.sqrt(1./(n-1)*sum(
            [math.pow(x-sample_mean, 2) for x in sample_list]))
    sample_standard_error = sample_standard_deviation/math.sqrt(n)
    return sample_mean, sample_standard_deviation, sample_standard_error

n_list = range(100, 50000, 200)
mean_list = []
sem_list = []
for n in n_list:
    mean, std, sem = run_experiment(n)
    mean_list.append(mean)
    sem_list.append(sem)

plt.figure(figsize=(8,5))
plt.subplots_adjust(hspace=0.5)
tmp_plot = plt.subplot(211)
tmp_plot.set_title('MEAN')
plt.plot(n_list, mean_list, 'g')
tmp_plot = plt.subplot(212)
tmp_plot.set_title('SEM')
plt.plot(n_list, sem_list, 'r')

```


## 3. CI (Confidence Interval) 置信区间
The standard error of our estimate is a measure of confidence. SEM标准误差就用作衡量测量的信心；标准误差越小，则 sample mean 越向 population mean 集中，置信度就越高。

### 3.1 定义
$CI=\widehat\mu\pm Z_{\alpha/2}\ \Large\frac{\widehat\sigma}{\sqrt{n}} \normalsize =\widehat\mu\pm Z_{\alpha/2}\ SEM$

- $SEM$：标准误差
- $Z_{\alpha/2}$：confidence coefficient 置信度系数
- $\alpha$ ：confidence level 置信度/置信水平

置信度常选择：90%、95%、99%，最常见的是 95% 置信区间，表示重复采样100次中有95次 population mean 在该区间内。

可以通过查表把置信度转化为置信度系数.例如：$\alpha=0.95, \alpha/2=0.475$，Z值表中对应 $Z_{\alpha/2}=1.96$，则相关置信区间为：$\widehat\mu\pm 1.96\ SEM$。

[置信区间通俗介绍](https://zh.wikihow.com/%E8%AE%A1%E7%AE%97%E7%BD%AE%E4%BF%A1%E5%8C%BA%E9%97%B4%EF%BC%88Confidence-Interval%EF%BC%89)

[Z值表](https://www.statisticshowto.com/tables/z-table/)

### 3.2 Z值计算
其实非常简单，就是计算 Normal Distribution 正态分布下，多少倍 SD（standard deviation）标准差的区间范围能覆盖相应置信的概率。

可以通过 PPF 函数求解 z值，详见下述代码：

- CDF Cumulative Distribution Function 累积分布函数
- PPF Percent Point Function 百分比点函数，是 累积分布函数 的反函数


```
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

alpha = 0.95
z = stats.norm.ppf(alpha+(1.0-alpha)/2)
print('z:', z)

mu = 0
std = 1
ci = (mu - z * std, mu + z * std)
x = np.linspace(mu - 4*std, mu + 4*std, 100)
y = stats.norm.pdf(x, mu, std)
plt.plot(x, y)
plt.vlines(ci[1], ymin=0, ymax=0.1)
plt.vlines(ci[0], ymin=0, ymax=0.1, label="95% CI")
plt.legend()
plt.show()
```


## 4. Hypothesis Test 假设检验
hypothesis test: is the difference in means statistically different from zero (or any other value)? 

### 4.1 分布之差
正态分布的性质有 两个正态分布的差仍然是正态分布：
$$N(\mu_1,\sigma_1^2)-N(\mu_2,\sigma_2^2)=N(\mu_3=\mu_1-\mu_2,\sigma_3^2=\sigma_1^2+\sigma_2^2)\\\sigma_3=\sqrt{\sigma_1^2+\sigma_2^2}$$

类似地，对于两组样本，他们的 sample mean 相对 population mean 是正态分布，所以有：
$$\mu_{diff}=\mu_1-\mu_2\\SEM_{diff}=\sqrt{SEM_1^2+SEM_2^2}$$

针对新的正态分布 $N(\mu_{diff}, SEM_{diff}^2)$，可以求得置信区间例如 $\mu_{diff}\pm 1.96\ SEM_{diff}$： 

- 如果其置信区间内不包含0，则表明两组样本的 population mean **有统计置信的差异**。
- **并且，新分布的置信区间能反映出两组样本绝对值上的置信差异**。

### 4.2 Z检验
The z statistic is a measure of how extreme the observed difference is. 假设检验用于衡量我们观察到数据有多极端（小概率）。都是反证的套路。

$$z=\frac{\mu_{diff}-H_0}{SEM_{diff}}=\frac{(\mu_1-\mu_2)-H_0}{\sqrt{\sigma_1^2/n_1+\sigma_2^2/n_2}}$$

- 引入 null hypothesis 原假设 $H_0$，通常 $H_0=0$，即假设分布之间是没有差异的
- 计算 $z=\mu_{diff}/SEM_{diff}$，这个 $z$值就是置信度系数
- 查表判断 $z$ 值对应的概率，即是原假设成立的概率
- 根据原假设出现的概率，接受/拒绝原假设

例如：当使用95%置信度，发现 $z=2.4>1.96$，则认为原假设条件下观察到数据概率太小太极端了（概率 < 5%），所以拒绝原假设，判定两个分布有置信差异。

### 4.3 假设检验 vs. 置信区间 
假设检验相对置信区间更松弛有弹性一些。某些情况下，两组样本的置信区间有重叠，但是却能有置信差异。

```
cont_mu, cont_se =  (71, 1)
test_mu, test_se = (74, 7)

diff_mu = test_mu - cont_mu
diff_se = np.sqrt(cont_se + cont_se)

print("Control 95% CI:", (cont_mu-1.96*cont_se, cont_mu+1.96*cont_se))
print("Test 95% CI:", (test_mu-1.96*test_se, test_mu+1.96*test_se))
print("Diff 95% CI:", (diff_mu-1.96*diff_se, diff_mu+1.96*diff_se))
```

### 4.4 p-value p值
The p-value is the probability of obtaining test results at least as extreme as the results actually observed during the test, assuming that the null hypothesis is correct. 

p值就是假设原假设成立（两组样本无差异）时，我们观察到相应数据的概率。
实际上直接根据 $z$ 值计算得出。可以更量化地衡量置信程度。




