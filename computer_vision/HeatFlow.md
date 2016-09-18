<font size=7 face='黑体'>序言</font>
在图像处理方面，我们从自然中汲取了大量灵感，例如射线、重力、水流等等物理学模型。其中最为常用的，似乎还是热力学模型，其抽象而简洁地描述了热传播、热均衡的过程。本文将对热力学模型展开细致讨论，不止揭露数学之美，更有自然之美。

<font size=7 face='黑体'>目录</font>

[TOC]



#1. 热力学模型

##热传递的规律
$$\Large    \frac{\partial T(x,t)}{\partial t} =\alpha \frac{\partial ^2 T(x,t)}{\partial ^2x}$$
热力学模型使用了上述公式对热传递过程进行描述。其中，$\alpha$ 表示热扩散速率，第一个简单系数，由介质材料决定； $T(x,t)$ 函数表示空间 $x$ 在 $t$ 时刻的整个温度分布，$x$ 可以为任意维度的空间。这个简单的公式，建立了温度分布在时间和空间上的直接关系；即空间中某一点的温度随时间的变化趋势由该点在空间中温度分布曲率直接决定。

<font color='#DA70D6'>这里举一个简单的热传递实例。</font>假设 $x$ 为一维空间，且 $T(x,t=0)=sin(x)$；即整个系统在初始情况下，温度在空间中正好是正弦分布。那么，整个系统的温度在下一个时刻如何变化呢？$\frac{\partial T(x,t=0)}{\partial t} =\alpha \frac{\partial ^2 T(x,t=0)}{\partial ^2x}=\alpha \frac{\partial ^2 sin(x)}{\partial ^2x}=-\alpha \ sin(x)$，变化方向正好与现在分布相反。也就是说，当前温度为正的区域温度将会下降，温度为负的区域温度将会上升，而且温度越高或者越低则变化幅度越大；这与我们的生活常识一致。热传递在一些复杂空间和复杂分布的系统中可以变得极其复杂，但是本质上却是非常简单的。

##热力迭代
了解热力流动的规律有什么意义呢？掌握了规律，我们就可以对整个热力系统沿时间轴进行推演。所有的热力学模型应用都基于此。
$$\Large T_{next}=T_{previous}+\Delta T=T_{previous}+\Delta t\frac{\partial T}{\partial t}$$
热力系统的迭代过程如上式所述。每一个时刻，我们求出热力系统中接下来的温度变化量，从而不停地把整个系统推向下一时刻。这里的温度变化量由时间差 $\Delta t$ 和温度变化率 $\frac{\partial T}{\partial t}$ 共同决定；前者是一个重要的模拟参数，决定系统迭代的步幅；后者则由之前的热传递公式求出。


##标准热力系统
一个标准的系统由3部分构成：

- Initial Condition
$T(x,t=0)=F(x)$

- Boundary Condition
$T(x\in boundary) = \Phi(x,t)$

- Heat Equation
$\frac{\partial T(x,t)}{\partial t} =\frac{\partial ^2 T(x,t)}{\partial ^2x}$

也就是说，除了热力学方程外（上述公式传导系数 $\alpha$ 被设为1而消去），我们还需要设置系统的初始状态，以及系统的边界条件。系统的边界条件有很多不同的选择，常用的是Dirichlet条件，就是让系统边界固定在一个温度，如零度上。
总的来说，热力学模型，就是在确定初始热力空间分布以及空间边界条件的情况下，通过热力传递公式，选定合适时间步幅，沿时间轴递推演进热力空间分布的过程。

#2. 常见滤镜
热力学在图片上最为常见的俩个应用为：
- isotropic diffusion (各向同性扩散)
- anisotropic diffusion (各向异性扩散)

这都是把图片的intensity(灰度值)视作初始温度，把图片边框的1像素固定为边界，边界温度维持不变；然后在图片上迭代推演热力传递公式，扩散热力，更新各个像素温度。（作为边框的1像素最后将被丢弃）

##各向同性扩散与高斯模糊
各向同性扩散完全等价于高斯模糊以及低通滤波。

###高斯模糊
高斯模糊是一种常用的去噪或者模糊图片方法，其具体方法是对原始图片和一个高斯模板进行卷积。核心在于高斯模板，这是一个至少3*3的奇数边长的矩阵，从中心点到四周数值呈现各向同性（即convariance matrix为值一致的对角矩阵）高斯分布；另外，模板所有值和为 1（normalize）。本文不作更多讨论。

由 $I*Template=F^{-1}\{F(I)F(Template)\}$，原图与模板的卷积等效于原图和模板在傅里叶空间中之间相乘并进行傅里叶逆变换；高斯模板在傅里叶空间中仍是一个高斯分布（标准差变倒数），傅里叶空间中的高斯模板将会强化低频信号并滤过高频信号，高斯模糊完全等价于低通滤波。

###等价的各向同性扩散
这里我们从数学上来证明这个等价。
首先注意到空间曲率在离散空间上的求解：
$$\begin{aligned}
\partial T(x)\ &=\frac{T(x+1)-T(x)}{1}=\frac{T(x-1)-T(x)}{1}\\
\partial^2 T(x)\ &=T(x+1)+T(x-1)-2T(x)
\end{aligned}$$这实际上也是拉普拉斯算子(Laplacian operator)，具体展开在二维空间的话：
$$\partial^2 T(x,y)=T(x+1,y)+T(x-1,y)+T(x,y+1)+T(x,y-1)-4T(x,y)$$所以，各向同性扩散的热力迭代可以展开如下：
$$\begin{aligned}
T(x,y,t=1)&=T(x,y,t=0)+\Delta T\\
&=T(x,y,t=0)+\Delta t\partial^2 T(x,y)\\
&=\Delta t T(x+1,y)+\Delta t T(x-1,y)+\Delta t T(x,y+1)+\Delta t T(x,y-1)+(1-4\Delta t)T(x,y)
\end{aligned}$$显然，这个迭代式表示的就是原图片与一个3*3的高斯模板进行卷积，其中 $\Delta t$ 应该小于0.25以确保迭代稳定。
##各向异性扩散与美颜滤镜
Anisotropic diffusion（各向异性扩散）是常用的图片加强算法。它在各向同性扩散的基础上前进了一步，在模糊图片的同时保持住图片中的边缘锐度，达到有选择的模糊。这一方法常见于美颜相机中，它可以保持住脸部的线条同时通过模糊消除斑点，让皮肤看起来整洁、细腻、柔和。另外，也是非常好的图片去噪方法。

### 算法原理
算法的关键是在扩散中保护边缘信息（即图片中物体的轮廓）。轮廓往往和其周围环境是有强烈对比度的，这种时候，其温度的空间梯度 $\frac{\partial T(x,t)}{\partial x}$ 将是一个较大的值。所以我们利用这一点来构造各向异性扩散，抑制高温差之间的热传递，以保护高温差代表的轮廓信息。

这里我们使用一个函数求取扩散抑制系数：
 $$\begin{aligned}
 \Large k(x)\ &=\Large g(\partial T(x))\\\Large g(x)\ &=\Large e^{-\frac{x^2}{q^2}}\end{aligned}$$ $x$为空间二维向量；$g(x)$是一个值域$(0,1]$的单调递减函数，所以温差越高，扩散系数 $k$ 越小。这里还有一个抑制敏感度参数 $q$。$q$ 越小，函数就对温差越敏感，即使很小的温差也触发明显的抑制；$q$ 越大，则越迟钝。我们的温度范围是 $[0,255]$，所以 $q$ 参数一般是[10,100]。

基于上一节推导的拉普拉斯算子，进行变化：
$$\begin{aligned}
\partial^2 T(x,y)\ &=T(x+1,y)+T(x-1,y)+T(x,y+1)+T(x,y-1)-4T(x,y)\\
\partial^2 T(x,y)\ &=G_n+G_s+G_w+G_e\\
G_n\ &=\partial T(x)=T(x-1,y)-T(x,y)\\
G_s\ &=\partial T(x)=T(x+1,y)-T(x,y)\\
G_w\ &=\partial T(y)=T(x,y-1)-T(x,y)\\
G_e\ &=\partial T(y)=T(x,y+1)-T(x,y)\\
\end{aligned}$$

最终，各向异性扩散的热力迭代可以展开如下：

$$\begin{aligned}
T(x,y,t=1)&=T(x,y,t=0)+\Delta T\\
&=T(x,y,t=0)+\Delta t  \partial^2 T(x,y)\\
&=T(x,y,t=0)+\Delta t \{ k(G_n)G_n+k(G_s)G_s+k(G_w)G_w+k(G_e)G_e\}
\end{aligned}$$

### Python代码实现
``` python
def anisotropic(image, time_gap=0.15, iterations=20, q=10):
    img = image.copy().astype(np.int)
    deno = q ** 2
    s = img.shape
    tmp = np.zeros(s, img.dtype)
    for i in range(iterations):
        for x in range(1, s[0] - 1):
            for y in range(1, s[1] - 1):
                NI = img[x - 1, y] - img[x, y]
                SI = img[x + 1, y] - img[x, y]
                WI = img[x, y - 1] - img[x, y]
                EI = img[x, y + 1] - img[x, y]
                cN = math.exp(-NI ** 2 / deno)
                cS = math.exp(-SI ** 2 / deno)
                cW = math.exp(-WI ** 2 / deno)
                cE = math.exp(-EI ** 2 / deno)
                tmp[x, y] = img[x, y] + time_gap * (cN * NI + cS * SI + cE * EI + cW * WI)
        img = tmp
    return img
```

#3 图像分割
热力学还有着更为复杂和有趣的应用，特别是在图像分割上，这一点后续再做整理




















