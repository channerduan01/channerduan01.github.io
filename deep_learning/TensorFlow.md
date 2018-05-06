TensorFlow
总的体感：TensorFlow以Tensor为Node，Operation为Edge，静态构建(lazy)整个Graph来定义模型。
Node：包含模型参数，静态变量，placeholder(预留的input接口)
Edge：计算操作，损失函数，Optimizer

最终我们会通过Session来run整个graph。

# Session
这是真正驱动模型或者Node计算的接口。


# Basic
Tensor 的定义和 Numpy 的 Array 有很多类似之处，重大不同是：
- Tensor 可以跑在GPU上
- Tensor 操作是，是lazy的，只会为Tensor对象保存操作，所有操作在后面编译并真正运行时才会触发；一个实例：np.zeros((2,2)) 近似 tf.zeros((2,2)).eval()

Feed

Fetch





# What is Embedding
更通俗的解释一下embedding 这个拓扑学的词儿。
地图就是对于现实地理的embedding，现实的地理地形的信息其实远远超过三维 但是地图通过颜色和等高线等来最大化表现现实的地理信息。

word embedding 也就是用固定的维度来最大化表现词的信息。
另外，Word embedding 是一个普适的概念，或者任务。为区别one-hot的词向量，可翻译成词嵌入。
Word2vec 是 Google 实现 word embedding 的一种具体的approach。




# TensorBoard 数据可视化分析
非常强悍的可视化工具，用于解析 tensorflow 运行过程中记录的特征格式log并可视化。

有命令行执行命令：“tensorboard --logdir=/Users/channerduan/logs/” 进行后台启动，然后浏览器登录：“http://localhost:6006/”进入可视化系统
这里有俩个重要的点：
- logdir 参数指定在windows上有bug，必须按照如上linux路径格式指定，所以如果文件不在“C:”，得先cd到相关磁盘下（如"cd E:"）再执行启动命令，否则tensorboard找不到文件（害我调试了半天...）
- 浏览器代理设置（proxy）中需要设置“跳过本地ip的解析”，否则用浏览器无法登陆可视化系统



