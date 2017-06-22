### 序言
Spark 是当前比较热门的大数据计算平台，常见与 Hadoop 的 Yarn 以及 HDFS 结合来完成大数据计算任务。  
Spark 相比传统的 MapReduce 模式，有如下优势：

- 大量适用于分布式计算的 operation；MapReduce 确实经典，不过相对繁琐
- 提供了 memory cache（号称加速读取100倍）；显然 cache 对 迭代算法 iterative 非常有利。
- 深度地支持更多语言：Java、scala、python、R（当然，scala 是直接编译字节码运行在 Java virual Machine 上，效率明显高于其他）；MapReduce 则是通过 Hadoop Streaming 这种间接方式支持的。
- 酷炫的 spark-shell，interactive 式地开发或查看数据



本文集合我对 Spark 理解、经验，来整体介绍一下 Spark。

# 1. Spark 架构
个人倾向于区分出物理架构以及运行时架构。物理架构是 Spark 集群本身的构成描述。而Runtime 架构是 Application 具体执行时的集群动态生成、分配的执行单元架构。此外，还有 Application 任务具体分解的结构（任务必须能被很平均地分解才能充分发挥集群计算的性能，所以很重要）。  

## 1.1 物理架构
物理结构描述了 Spark 集群的物理构成。
<center>
<img src="http://img.blog.csdn.net/20170614120215055?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQ2RkMnhk/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" width="80%" height="80%" />
Figure 1. Physical Structure of Spark
</center>

### 1.1.1 Driver Program（Client）
本地启动程序。我们提交 spark-submit 或者 启动 spark-shell 的时候就是创建了一个 driver 用于执行 Application。  

Driver 负责创建并持续维护本地基本执行环境 SparkContext（1.2中详解）、HiveContext 等，并通过我们配置的参数来访问集群。Driver 会和 Cluster Manager 提交任务，申请执行资源；之后与 Worker 上的 执行单元 Executor（1.2中详解，相当于申请到的计算机资源）进行交互，持续发布具体任务（切分好的 Task）、跟踪执行、获取结果。

### 1.1.2 Cluster Manager
集群管理节点。管理内容包含 内存、CPU 等资源划分以及 任务提交队列 的管理；例如通常是通过一个FIFO队列来管理大家的 spark 任务提交，分配有限的资源。当任务需要的资源准备就绪，Cluster Mnager 就会告知 Driver 具体的资源细节（比如那个机器那个端口），之后 Driver 就可以直接与 Worker 通信，推送代码并开始跑任务。  

集群常见的部署方式可以是基于 Yarn 或者 Standlone；前者为 Hadoop 平台通用的资源管理框架，后者为 Spark 直接负责管理集群资源。另外，集群底层的分布式文件存储一般使用 hdfs。而 Spark 集群本身专注于 分布式计算，搞定自己的核心任务就好。

### 1.1.3 Worker Node
集群执行节点。一个真正的物理节点可以虚拟出多个 Worker，其负责执行真正的运算。Worker 由 Cluster Manager 管理，任务执行时由 Worker 上会创建 Executor（也被称为 Container 容器） 进程实例执行具体代码。Worker 负责管理、监控其创建的 Executor，必要的时候可以直接 kill 掉。

注意，Worker 只对 Cluster manager 负责，不与用户侧 Driver 交互。而 Worker 运行时创建的 Eexcutor 与 Driver 直接交互。

## 1.2 Runtime 架构
Runtime 结构指的是 Spark 具体执行 Application 时，在集群上动态创建出来的执行体的结构；包括 Driver 上的 SparkContext 和 参与执行的 Worker 上的 Executor。他们的生命周期都贯穿整个 Application。

<center>
<img src="http://img.blog.csdn.net/20170614120303665?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQ2RkMnhk/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" width="80%" height="80%" />
Figure 2. Runtime Structure of Spark
</center>

### 1.2.1 SparkContext
由 Driver 创建、维护的 SparkContext，用于与整个集群的 Executor 进行直接交互。它可以直接创建 RDD、广播共享变量等等，具体包含了 RDD Graph、DAGScheduler、TaskScheduler等模块。这里面包含着很多重要的元数据，例如各个 RDD 之间的关联信息，所以集群数据丢失时可以精确到 partition 地自动地高效地恢复丢失数据；以及 Hive 表的 schema 信息等。

这里引出了一个潜在瓶颈：由于 SparkContext 职责重大，对于数据进行着 partition 粒度的精确追踪记录；当数据量过大时，SparkContext 本身可能会卡死。

### 1.2.2 Executor
Executor 是由 Worker 创建的具体执行单元，拥有独立的 JVM 进程，又被称为容器 Container，在一个隔离独立的环境中执行 Task。Executor 在初始化时会载入所执行 jar 包以及相关依赖。

Executor 被 Worker 监控，但会直接与 Driver 的 SparkContext 进行交互，接受并执行具体任务（真正工作在一线，执行我们的代码）；相关 RDD 的 cache 也是直接保存在 Executor 的内存中。

我们在提交任务的时候可以指定每个 Executor 的计算资源数量（cores）以及内存资源数量（多少gb）。core 的数量（默认1）决定了 Executor 内部的并发线程数量，有 n 个 cores 的话就可以并发执行 n 个 Task；所以计算资源上整体的并发能力是 Executor数量 * 每个Executor上cores数量。

## 1.3 Application 执行结构
如果我们想达到理想的高并发执行，上述 1.2 提到的 Executor数量 仅是提供了 potential，我们任务 Application 本身是否能够被充分并发才是性能的关键。这就要涉及 Application 的分解执行结构。

### 1.3.1 Job
Spark 代码中的 action 会触发一个 Job。Job 本身会由 SparkContext 在 DAG（Directed Acyclic Graph）上优化并规划出具体执行方式。
### 1.3.2 Stage
每个 Job 都会被划分为一系列阶段来执行，比如说最经典的 Map-Reduce 就是俩个不同阶段。通过 DAG 的优化规划，不同的阶段是可能并发执行的。
### 1.3.3 Task
Stage 会被切分为最细粒度的执行任务 Task，Task 由 Executor 直接执行，一个 Executor 上的 cores 数量决定了它能并发执行的 Task 数量。

**Task 的数量极其重要！** 数量不够的话，有以下问题：

- 即使有再多计算资源，也只能闲置；因为任务无法切分开
- 单个 Executor 处理的任务过于庞大，内存上可能会直接挂掉（单个 Executor 通常只分配 几G 内存）

对于原始数据，Task 的数量通常由 数据 RDD 的 partition 数量决定；数据处理过程中的 reduce 以及 sort 等操作可能会急剧缩减 partition 数量；所以我们有时需要使用 repartition 强制把数据 shuffle 开，扩大 partition 数量来保证后续的 Task 数量。

# 2. Spark 底层
这里展开介绍一些 Spark 的核心底层模块以及机制，包括核心的数据对象 RDD 和 其支持的两类算子 operation： Transformation 和 Action，以及一些其他重要机制。

## 2.1 RDD
Resilient Distributed Dataset，Spark 统一使用 RDD 对象来抽象地表示一份存放在集群上的数据。以下分开描述几个其关键特性：

### 2.1.1 Resilient
这一数据对象是弹性的，这主要表现在其所有的 partition 都被 Driver 上的 SparkContext 精确跟踪记录，这是通过 DAG（后面再展开讲解）实现的。所以在相对不稳定的集群环境中，RDD 的任何 partition 丢失都能被 Driver 追溯到，以最小的代价进行自动恢复。

### 2.1.2 Distributed
RDD 具体会切分成不同的 partition 分布在集群上，就如同 HDFS 的文件切分成 block 分布在集群上，并且 RDD 还可以像 HDFS 上的文件以一样设置备份。他们确实有点类似，但 RDD 作为 Spark 计算引擎的数据基础，提供了非常丰富的分布式计算接口、功能，以及进一步的抽象（如把 Spark SQL 把 RDD 封装抽象成 DataFrame，Spark Streaming 把 RDD 封装抽象成 DStream）。

如 1.3.3 所述，partition 的数量也直接决定任务能够被分解成几个 Task 来并发执行，对任务执行性能非常关键。

### 2.1.3 Dataset
任意的 RDD 都是一个静态的常量，是  immutable 的，可以理解成 scala 中的 val 或者 java 中的 final。我们无法去修改一个 RDD 的某一部分，只能通过 transformation 把当前 RDD 转换到另一个 RDD 来完成修改。这种限制有点像 hive 的限制，我们倾向于对大数据只进行读操作来搞数据挖掘，而不是频繁修改更新（这应该是业务系统使用数据库做的事）。另外，这种限制，也使得 Spark 对 RDD 的追踪更加的简单，更好地完成 Resilient 的工作。

## 2.2 Transformation
2.1 中提到了，RDD 只能通过 transformation 来变成另一个 RDD，也就是说我们队大数据进行的操作只能通过是 Spark 提供 transformation 实现。当然，相比于经典的 Map-Reduce，Spark 提供了数十个算子 operation，功能非常强大。

### 2.2.1 lazy
Transformation 的所有 operation 都是 lazy 的，相关代码被执行后，只会在 DAG 中生成一个标记，不会立刻对数据执行相应的操作；真正的数据操作只会在最终 Action 被调用后再执行。这有点像 tensorflow，其也是创建一个数据处理流程 Graph，在数据处理流程定义完毕后进行编译优化，之后才能执行。这可以极大的提升计算性能，后面会展开谈一下 DAG。

### 2.2.2 narrow & wide
####Transformation 可以分类为 narrow 和 wide 两类：

- narrow，操作可以在当前节点内完成，是 local 的，所以性能会比较好。例如 flatMap、Map 和 filter 都是 narrow 的，它们把当前 partition 内的 element 逐个映射处理，转换出新的 RDD 的 partition。
- wide，操作需要不同节点 shuffle 完成，这种操作需要集群参与，代价显著提高。shuffle 是极其消耗性能的操作，而且按照处理逻辑 shuffle 后的数据可能会倾斜、不均匀，这会为后续的处理流程再次带来障碍。例如 groupByKey 和 reduceByKey 都是 wide 的，都需要相同的 Key 的信息 shuffle 到一个节点下面，完成最终处理。

####这里额外提一个有趣的点，同属 wide 操作性能差异也能很大
**groupByKey & reduceByKey**：

- groupByKey 开销很大，它需要把所有相关数据都 shuffle 到一起（key统一），再把所有value打包存在这个key下面；可以接受自定义处理函数，来处理(key, iterator) 类型的数据。
- reduceByKey 相对就轻量很多，因为每个key只产出一个最终值且不需要数据放在一块计算，可以各个机器上先直接执行 reduce 相关函数，然后再 shuffle 合并计算结果；也就是说 reduceByKey 只 shuffle 计算结果 而 groupByKey 要 shuffle 原始数据。当然，reduceByKey 也额外要求自定义函数 associative（结合的） 且 commutative（交换的），所以我们可以任意 merge 中间结果。

### 2.2.3 Partition 控制
之前已经强调过多次，RDD 本身的 partition 数量，其在集群上的 均分分布程度对计算性能至关重要。在相对复杂的应用中，除了 Spark 自动控制 partition 外，我们使用 coalesce 和 repartition 算子来重新分割数据是非常有必要的，可以实际中解决很多 crash 以及性能瓶颈，这里单独强调一下这俩个 算子。
#### **coalesce**
单词本身是合并的意思，这是一个轻量级的 partition 合并算子，传入的参数必须小于 RDD 当前 partition，否则自动忽略不执行。这个算子可以近似看做是 narrow 的，即 Spark 会把操作尽量局限在 local，不触发代价较高的 shuffle。

实践中，我一般在计算结果导出（例如计算完了写入 HDFS 或者 HIVE表）的时候会先进行一个 coalesce 来防止数据以大量小文件形式写入集群。或者有时候原始数据集实在太大（10TB级），原始分区 partition 也很大，直接对 Driver 的 SparkContext 造成很大压力（跟踪所有分区）；所以使用 coalesce 消减分区规模。

#### **repartition**
重分割 repartition 是标准地将 RDD 数据重新分割，分割成多少份都可以，会强制触发 shuffle，把数据在集群上均匀分布，所以性能开销非常大。当然，必要时候，特别是 RDD 的 partition 过小的时候还是必须得用，来保证后续处理流程中足够的并发数。


## 2.3 DAG
Directed Acyclic Graphs，这里独立强调一下支撑 RDD 的核心计算机制。

### 2.3.1 支撑 Resilient
之前有提到过 所有的 RDD 都是静态的常量，我们只能通过 transformation 转换出另一个 RDD 而不能直接修改 RDD。其实 RDD 就成为了的一个个节点，而 transformation 就对应相应的边，它们形成了一个有向无环图 DAG 的关系来表达出整个数据处理的流程。这一切被存放在 Driver 的 SparkContext 的 RDD Graph 中，所以 Driver 能够精确地追踪整个数据处理流程中的任何一个数据分片的由来，直接支撑了 Resilent 的关键特性。正是因为 DAG，RDD 才能被称为 RDD。

### 2.3.2 自动优化
DAG 的另一关键点在于，因为 Spark 掌握整个数据流的处理细节，而且这些处理细节都是 lazy 的还没有执行；所以当 action 真正触发 Spark 去执行 DAG，Spark 有很大潜力去自行组织更优化的流程来分布式执行 DAG。也就是说 step by step 的一串 transformation 和 临时 RDD 只是我们自己的抽象定义，Spark 真正执行的时候可能一步到位，中间环节都不存在的，毕竟我们只要最后 action 得到的那个结果。这里列举几项典型优化：

- 不同 Stage 可以并发。例如可以并发地计算俩个不同 tables，之后再将他们 join，这就充分利用了集群资源。
- 加速 narrow operations。例如我们连续使用多个filter以及map来过滤加工本地数据，但是实际上 Spark 没有经历那么多步骤去处理数据，它只是遍历了一遍数据，对于每一条数据直接执行一系列的操作。
- 调整计算次序。例如 Spark 可以把 filter 算子放在 map 算子前面来缩减后续计算规模。

以上是简单举例，Spark 底层具体如何优化取决于具体的内部策略了。

## 2.4 Action
RDD 支持的算子 除了 Transformation 就是 Action；当然 Action 要简单的多。Action 会触发真正的计算，让 Spark 创建一个 Job，真正去规划（切分 Stages、Tasks）执行之前一系列 Transformation 定义的 DAG，**然后把结果返回到 Driver**。
最有趣的地方是，数据载入、缓存、处理的全部操作基本都是 lazy 的，即使 Spark SQL 载入数据库表也是先只载入 表结构 schema；一切流程都是由 Action 驱动才开始执行的，缓存 cache 操作一般会拆分为相关的一个 Stage。

### 2.4.1 一些 flavor
这里简单列举一些常用的 Action，感受一下这和 Transformation 是多么不同。

- 偏调试使用：collect 将所有数据以 Array 形式返回给 Driver，数据量大 Driver 肯定会爆；count 统计整个 RDD 的 element 总数；first 返回 RDD 一个完整的 element；take(n) 返回 n 个 element，相应的 Spark SQL 中常用 show(n) 返回 n 行表内容
- 保存结果：saveAsFile 将 RDD 内容以 element 为行保存到文件（直接保存到 HDFS）

以上只是简单给出一些 Action 的感觉，当然，Spark 内部对 Action 也有优化，例如只是用 first 或者 take 的话，使用者只关心一个很局部的结果，所以 Spark 不会对全量数据集展开计算，可能只计算一个对应的 partition 就直接返回 Action 了；相应的，使用 count 的话 Spark 就必须完成全量计算了。

### 2.4.2 额外的话题，reduce
Action 里头包含了一个 reduce 算子，我很少用到，但是这个东西和经典的 hadoop 的 Map-Reduce 是完全不同的，这里可能会有一点混淆。Map-Reduce 中的 "Reduce" 比较类似 Transformation 中 groupByKey，它会把 Key 相同的数据 shuffle 到一起，然后使用用户定义的流程去处理各个 Key 相关的数据。2.2.2 中也讨论过 reduceByKey 性能高于 groupByKey 因为前者可以直接把 value 在各个节点上 reduce 到1个再 shuffle，而后者是把 value 全部 shuffle 到一起后再处理的。可以搜到经典的 Spark 的 Helloworld (WordCount) 使用的就是 reduceByKey。最后，Spark Action 的 reduce 其实类似 count，它没有 Key 的概念，而是通过自定义函数 (T, T) => T 这种形式直接压缩 value ，也就是说 RDD 执行 reduce 后最终只返回一个 element 结构，完全不同。感觉 Hadoop 的 reduce 更接近 Spark 中的 reduce 或 reduceByKey 或 groupByKey 是一个很好的面试题！

## 2.5 cache
内存 cache 是 Spark 的核心特性，一个非常大的优势就是把经常用到的数据 cache 到内存中来大幅节省频繁读写硬盘的性能消耗。这里专门列出来强调一下这个话题，给出几个 Tips：


- 不要 cache 原始数据，对数据完成裁剪、清洗之后再 cache；珍惜内存。
- cache() 和 persist() 是等效的，也都是 lazy 的。之前也提到，Job 被 Action 触发执行后，流程中 cache 往往会作为一个 Stage，cache 执行完毕后，再执行针对这个 cache 数据集 的下一步操作。也是因为这个原因，我们调试时可以考虑在代码中加入一些 cache 来查看哪个阶段会比较耗时。
- unpersist() 可以用来干掉已经 cache 住的数据，要注意的一点是这个代码可能会立刻执行（取决于 Spark 本身策略），会导致数据立刻丢失，所以要确保相关 RDD 之后不会调用了，否则要重新再计算一遍。
- 在 Spark 的任务监控网页上有一个 Storage 标签，可以查看到我们所有 cache 住的数据详情，包括 RDD的名称、数据的大小、相关 partition 数量、cache 住的比例等关键信息。

# 3 相关经验
我目前主要使用 Spark 核心库以及 Spark SQL 的功能，主要是离线分析大数据，这里记录一些实践中的具体经验。

## 3.1 细节技巧

### 3.1.1 调起 Spark
一般分为 spark-submit 和 spark-shell 两种模式。
#### spark-submit
这是常用的部署方式，把任务相关代码提交到集群进行执行，我常用到的完整提交指令用例如下：
"spark-submit --class com.minerva.grain.ScoreEngine --num-executors 100 --executor-cores 2 --executor-memory 4g target/scala-2.10/minerva-grain_2.10-1.0.jar"

列举几个常用的配置参数：
- num-executor      标识我们申请的执行单元数量
- executor-cores    标识每个执行单元的CPU资源数量，即线程数量
- executor-memory   标识每个执行单元的内存大小（太大的话，可能会影响其他人使用；过小的话单个core分配到的内存可能太小而挂掉）
**这里要注意一点：我们的 jar 包一定要放在最后指定，否则前面配置的参数不会生效**
#### spark-shell
这是常用于调试的命令行模式，就是类似 matlab 命令行的东西，本地的 shell 会一直保持和集群的连接，方便我们随时把当前环境的一些变量打印出来，以及执行任意命令。调起 shell 的一个用例如下：
"spark-shell --conf spark.driver.maxResultSize=4g --num-executors 500 --jars /home/XXX/scala/target/scala-2.10/antifraud-minerva-devicerank_2.10-1.0.jar"

另外，一般 shell 会以如下形式展示 Application 的某一 Job 的某一 Stage 进度（Stage 的 index 为整个 Application 范围内的）：
Stage 46:=================>   (50+10 / 79)
其中 79 为当前 Stage 的总任务数，50 表示已经完成的任务数，10表示正在运行的任务数。79-50-10=19 的剩余任务，可能是由于任务依赖而没有执行，或者是执行资源如 num_executor=10 的限制而没有并发。

### 3.1.2 Spark SQL
实践中我比较喜欢使用 Spark SQL  相关的接口来操作大数据，这块把 RDD 封装成了 DataFrame 的形式，和 python 知名的 pandas 库很类似，Spark SQL 对 DataFrame 提供了一些列的结构化数据操作方法，例如：

- 不同 DataFrame 之间的 join 操作（这块 3.2.1 还要深入分析）
- 增减 Column；增加可以通过 UDF（User Defined Function）直接根据当前数据创建，以及用 lit 创建静态数据列；删除直接 drop 或者 select 忽略掉就可以
- groupby 拥有的大量内置的 聚集函数 Aggregate Function，并且支持 UDAF（User Defined Aggregate Function）；UDAF 可以自定义 input、cache、output 的类型，自定义算子 initialize（初始化 cache 中的各种数据结构）、update（本地reduce一个input）、merge（节点间合并结果） 和 evaluation（reduce完后计算出 ouput），非常灵活。

DataFrame 和 RDD 也都是可以互相转换的，而且 DataFrame 还能 register 到 SparkContext 中成为一个 virtual table，然后通过 Spark SQL 用纯正的 SQL 语句进行访问操作。

另一个非常方便的特点是，Spark SQL 能够直接读取 hive 中的数据为 DataFrame，并且处理完成直接后，可以把一个结果 DataFrame 直接输出到 hive，使得整个大数据处理流程非常简洁。

### 3.1.3 其他小细节
- flatMap数据清洗：在进行解析并清洗原始数据的过程中，可以使用 flatMap 承担数据解析+过滤 的双重功效。因为 flatMap 返回的结果可以 0到多个，我们对彻底解析失败的结果直接返回空数组，对解析成功的结果返回长度1的数组。

- UDAF：在使用 自定义聚集函数 UDAF 的时候，要特别注意 update 有很能会引入 null 的输入，这种输入应该直接过滤掉，必须所有逻辑外套一层 if (input.isNotNull)。

## 3.2 性能调优

### 3.2.1 join
join 操作很常用，这是一个很重型，应该很小心的操作。Spark SQL 默认的 join 方式是 Shuffle Hash Join，即把俩张表 key 相同的数据都 shuffle 到一个节点上，然后再来做 local 的 join 操作。如果数据倾斜严重 skew，那这样的 shuffle 操作将会把整个任务卡死。这里有个优化方法是我们对于 大表 和 小表 之间的 join 可以采用 Broadcast Join 调优，就是把小表 广播 Broadcast 到整个集群上，然后再 join，其实 hive 也有类似的方法。具体实现就是在调起 Application 的命令中加一个配置 --conf "spark.sql.autoBroadcastJoinThreshold=100*1024*1024"，上述的 threshold 设置不大于 100MB 的表我们使用 Broadcast Join。具体我们的小表有多大可以 cache 住然后查一下，当然太大了也不适合用 Broadcast Join。

## 3.2.2 partition 数量缩减
在整个数据处理过程中，大部分情况下 partition 数量控制我们是交给 Spark 的。但是 Spark 的一些操作，例如 sort 和 join 可能会导致 RDD 的 partition 数据突然缩减，特别是对一些严重 skew 的 key 进行 sort（例如这个 key 90%的值都一样）的时候。

partition 数量缩减会严重影响后续的执行效率（影响了并发能力），甚至一些节点会因为处理的 partition 的体积太大而直接 OOM out-of-memory 挂掉。所以我们要对 partition 数量，其实也能对应 Application 跑的时候各个 Stage 的 Task 数比较敏感，注意监控。必要时候使用 repartition 来恢复并发能力，当然这个操作代价也是非常巨大的。

## 3.3 DEBUG
这里简单列举一些我遇到过的 bug，我只用 scala 写 Spark 应用，所以一下都是 scala 跑的时候遇到的问题。（并且我使用的集群是 spark 1.6）

### 3.3.1 序列化失败
我们创建对象（new 出来）的话，一般的对象是无法序列化的，必须要实现序列化接口（相关 class 继承 Serializable）才能序列化。new 出来的对象放到 UDF 或者 UDAF 函数中参与计算的话，动态创建出来的对象无法发送到集群，就会出现 序列化失败 的问题（IDE 的编译的时候是检查不出来这种错误的）。不要 new 或者 给对象加上序列化可以解决这个问题。

### 3.3.2 Driver 造成的任务卡死
一般任务卡死跑不动都是集群上的节点卡主了，很可能是数据倾斜导致某些节点的数据处理量远远超过其他节点，然后把整个任务卡住了。但是某些时候 Driver 可能也会遇到瓶颈把自己卡死。我遇到过的情况是原始均匀分布在 HDFS 中的一个超大数据文件（TB级）无法载入，即使我申请数百节点（TB级内存）却无法完成简单的数据清洗载入，Application 一启动就卡住，直到超时挂掉都跑不出进度。这是由于 Driver 追踪上万 partition 压力太大，最终在数据读取操作后增加 coalesce 压缩 partition 到千量级解决问题。 



