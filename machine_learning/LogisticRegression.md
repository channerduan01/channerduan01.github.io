Logistic Regression



# Logistic (或 Sigmoid) 函数推导
这里将详细解释从 伯努利分布（Bernoulli Distribution）到 广义线性模型（General Liner Model）对 Logistic 的推导过程。



## Generalized linear model 广义线性模型
In statistics, the generalized linear model (GLM) is a flexible generalization of ordinary linear linear regression that allows for response variables that have error distribution models other than a normal distribution. response variable 响应变量指的就是模型的输出结果y。如果y本身服从Gaussian Distribution，我们直接使用linear model就可以了，但是，例如遇到二分类问题，y应该服从Bernoulli Distribution，这时我们必须使用广义线性模型的视角，通过link function



link function






广义线性模型 GLM（General Linear Model）

指数分布族 Exponential Distribution Family

连接函数 link function


LR 的 link function 又称为 logit

logistic回归是概率模型，非线性表达式，其线性表达式即logit回归。logistic回归计算的是P,而logit回归计算的是logit(p)。logit（p）=log(p/1-p)


GLM相关资料：
https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf
http://blog.csdn.net/dream_angel_z/article/details/46288167
http://blog.csdn.net/u011467621/article/details/48197943
http://www.cs.cmu.edu/~tom/10601_sp08/slides/LogRegr-2-11-2008-full.pdf
http://blog.csdn.net/lilyth_lilyth/article/details/10032993
https://www.slideshare.net/jaisalmer992/cd-driveprml2-4-d0703?utm_source=slideshow02&utm_medium=ssemail&utm_campaign=share_slideshow
https://www.zhihu.com/question/28469421
https://www.zhihu.com/question/36714044
https://www.zhihu.com/question/24429347
http://cs229.stanford.edu/notes/cs229-notes1.pdf
http://open.163.com/movie/2008/1/E/B/M6SGF6VB4_M6SGHM4EB.html
http://open.163.com/movie/2008/1/E/D/M6SGF6VB4_M6SGHKAED.html


EM相关资料：
http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html


树模型：
http://www.tuicool.com/articles/BRFremI

Tensorflow：
https://github.com/MorvanZhou/Tensorflow-Tutorial/tree/master/tutorial-contents













