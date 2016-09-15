#主成分分析法（Principal Components Analysis）

##求解主成分
主成分是指，对于原始数据矩阵 $X$ 方差最大化的一个投影矢量 vector。设 $X$ 的协方差矩阵（covariance matrix）为 $C$。我们这里利用 拉格朗日乘子 来求解主成分。

这是一个 带约束优化问题（constrained optimization problem）：  
###$\max \limits_{u} u^tCu,\ \ subject\ to\ u^tu=1$  

$u^tCu$ 是原始数据 $X$ 在$u$投影后方差值；这里的约束是由于我们想通过方向来最大化方差值，而不是投影矩阵的量级本身。 

$L(u,\lambda)=u^tCu-\lambda(u^tu-1)\\
\displaystyle\frac{\partial L}{\partial u}=2Cu-2\lambda u=0\\
Cu=\lambda u$  

满足上式的向量 $u$ 被称作特征向量，相关的 $\lambda$ 被称为特征值。特征向量相互正交（orthogonal），特征值表示原始数据在其特征向量上映射后的方差

协方差矩阵 $C$ 的特征值（eigenvalues）最大的一系列特征向量（eigenvectors）被称为 主成分（principal componentes）。



##PCA 的对偶求解
