#Hadoop

##Basis
###局限性
**整个大数据处理的前提是，数据是静态的；如果数据需要经常地、甚至是有时效性要求地更新，或者快速的检索查找，那么使用Hadoop这类型的大数据架构师不适合的（应该使用传统数据库，使用模式、索引...）。Hadoop更多地用于对大体量静态数据，建立Data Warehouse(DW)数据仓库，来做数据挖掘。核心在于“大”，能够处理大规模数据量，但是处理手段相对简单！**

###Define
Apache Hadoop is an open source software framework for storage and large scale processing of data-sets on clusters of commodity hardware. Most written in java and include some native-c and shell.
###Philosophy
- **Moving computation to data instead of moving data to computation in traditional way**
- Big data + Simple algorithm vs. Small data + Complex algorithm (The first may be better)
###Key Characteristics
####Scalability
Really for big data with **low cost**
####Reliability
All the modules within the hadoop framework are designed with the fundamental assumption that hardware fails.
####Flexibility

##Hadoop Stack, 其主体结构为以下1-4个层次
###1，Hadoop Common
Hadoop 框架的基础支持模块, libraries and ultilities

###2，HDFS (Hadoop Distributed File System)
这是大数据的基础，利用大量廉价存储资源，构造出分布式存储环境。这种大规模集群上，很容易出现失败，会有很多容错设计考虑在内。存储的文件也常会有 Data Replication，这样即使某些廉价节点挂掉，也不会丢失数据。

####第一代设计
- NameNode
唯一的中心节点，存储数据索引，指示出具体的数据被切片(blocks)存放于哪个机器，对于大数据储存，这个索引表可能会非常非常大。另外，一般会有 Second NameNode 随时对 NameNode 做 snapshot，来保证系统在 NameNode 突然挂掉时，也能正常运行。
只有一个NameNode，所以scalability会有问题~
- DataNode
具体的数据存放节点；数据是切片(block)后存放的，方便进行跟踪管理。

####第二代设计 HDFS Federation
Multi NameNode Servers + Multi Namespaces + Block Pools(DataNodes)
（总之就是多个NameNode~）

####一个核心的性能问题：文件切片大小（block size, default is 64MB）
这个问题也涉及到大量小文件对 hdfs 的影响
- NameNode 把每一个block视作一个对象进行追踪，大量小文件会产生大量block（算上replication影响更大），严重消耗 NameNode 的存储资源，并且可能对网络负载（Network load）造成压力（NameNode检查各个block状态）
- MapReduce 过程中，每个 block 会对应一个 Map 操作，小文件会产生大量任务；任务切换、小数据量的IO读写都会造成性能的很大浪费（所以IT领域general的系统优化思路是合并小数据为大数据块，然后一次性处理~）

####系统级别的重要参数
- block size，default 64MB，这个上面已经讲解过，对系统性能有重大影响，核心参数
- replication，default 3，冗余备份，表示包括数据本身一共会存储几份数据；这是 cost 和 robust 之间的权衡~ 另外，多重点备份的话，也有可能提升性能（数据离使用者更近）

- Name Quotas : 限制某个目录下的文件和文件夹数量；Space Quotas : 设置某个目录的空间大小

####HDFS Commands 使用心得
- ls、lsr、mkdir、du、rm、rmr、cat、mv、cp、tail 等命令和 linux 通用
- fsck 查看文件状态
- 文件导入（导出），put 和 copyFromLocal 基本一致，唯一的差别是后者强调了本地路径，而前者可以对本地路径或者hdfs路径操作；后者在某些时候可以避免路径歧义。（get 和 copyToLocal也类似）


###3，YARN（也被称为 execution engine）
负责 Cluster Resource Management，作为HDFS上的一个通用框架，用于协调管理整个集群资源。相当于在计算和存储之间，加入了一个集群管理的层次。这个层次是通用的，与具体计算方法无关，不局限于 MapReduce 数据处理，还可以服务于其他大数据处理框架，比如 Spark（Spark可以脱离YARN直接架在HDFS，也可以架在Hadoop之外的平台上），Machine Learning 常使用到 Spark。  

由一个中心节点 Resource Manager（会有相应的热备份节点，不然这个挂了全部计算都完了...） 以及一系列的处理节点 Node Manager 构成。中心节点可以接受客户的数据处理请求和子节点的资源申请请求，来统一协调完成任务。这一设计，分离了资源管理与数据应用 Application。每一个数据处理请求会生成一个 Application Master（例如作MapReduce的JobTracker） 运行在一个普通处理节点上，有需要的话，这个 Master 再向 Resource Manager 请求分配其他节点 Container（例如作MapReduce的TaskTracker）；多个任务可以并发执行。  

Scheduling 是很重要的一部分，我们可能会同时运算不同优先级任务；具体的资源管理是可以配置的，例如 Default 是 FIFO（First In First Out），或者指定不同用户、不用应用的资源限制，或者Fairshare 等等（当然，也不限制于YRAN，可以在整个存储、运算体系的各个层次上都涉及资源分配）

###4，MapReduce
Generally speaking it is a design pattern for big data! 把相对简单的数据处理任务，抽取为Map，Reduce俩个步骤，由平台完成核心的Shuffle过程。最大的局限是不是所有任务都能这么切分的；另外，MapReduce高度依赖IO，每次都从disk读取数据，这个是一大bottle neck。

微观来说，在Hadoop 2.0上，这基于 YARN（以及HDFS） 的大数据处理工具。核心思想在于：把处理逻辑配置到数据上。这一点对分布式存储非常重要，传统数据处理算法考虑的是一个整体的数据源，而大数据是存储在N个机器上，处理流程并发地Map到数据存储的机器进行处理是高效的方式（Move Computation to Data）  

这一层的逻辑和 HDFS 是完全分离的；不过类似的，其处理从中心节点（Master Node） JobTracker 发起，Map 到各个数据节点 (Slave Node) 执行 TaskTracker；其实就是把逻辑推出其要处理的数据上。

#### 一些基本概念
- 默认来说，一个mapper处理一个data切片（block）；当然，这个和block size以及设置有关
- 默认来说，建议one reducer per core；reducer的数量取决于机器CPU数量，保证并发执行
- 对于某些任务，可以使用 Cascade(Chain) Map/Reduce Jobs, 来复用一些结果，提升效率
- mapper后面可以加入一个combiner，部分承担reduce的工作，在某些任务中减少shuffling & grouping阶段的压力，提升整体性能
- 可以考虑使用bin key into ranges(对key分桶)来减少shuffling过程的压力，不过会增加reduce阶段的压力，这是一个针对实际情况的tradeoff

##### Map过程： 
1．每个输入分片会让一个map任务来处理，默认情况下，以HDFS的一个块的大小（默认为64M）为一个分片，当然我们也可以设置块的大小。map输出的结果会暂且放在一个环形内存缓冲区中（该缓冲区的大小默认为100M，由io.sort.mb属性控制），当该缓冲区快要溢出时（默认为缓冲区大小的80%，由io.sort.spill.percent属性控制），会在本地文件系统中创建一个溢出文件，将该缓冲区中的数据写入这个文件。

2．在写入磁盘之前，线程首先根据reduce任务的数目将数据划分为相同数目的分区，也就是一个reduce任务对应一个分区的数据。这样做是为了避免有些reduce任务分配到大量数据，而有些reduce任务却分到很少数据，甚至没有分到数据的尴尬局面。其实分区就是对数据进行hash的过程。然后对每个分区中的数据进行排序，如果此时设置了Combiner，将排序后的结果进行Combia操作，这样做的目的是让尽可能少的数据写入到磁盘。

3．当map任务输出最后一个记录时，可能会有很多的溢出文件，这时需要将这些文件合并。合并的过程中会不断地进行排序和combia操作，目的有两个：1.尽量减少每次写入磁盘的数据量；2.尽量减少下一复制阶段网络传输的数据量。最后合并成了一个已分区且已排序的文件。为了减少网络传输的数据量，这里可以将数据压缩，只要将mapred.compress.map.out设置为true就可以了。

4．将分区中的数据拷贝给相对应的reduce任务。有人可能会问：分区中的数据怎么知道它对应的reduce是哪个呢？其实map任务一直和其父TaskTracker保持联系，而TaskTracker又一直和JobTracker保持心跳。所以JobTracker中保存了整个集群中的宏观信息。只要reduce任务向JobTracker获取对应的map输出位置就ok了哦。

到这里，map端就分析完了。那到底什么是Shuffle呢？Shuffle的中文意思是“洗牌”，如果我们这样看：一个map产生的数据，结果通过hash过程分区却分配给了不同的reduce任务，是不是一个对数据洗牌的过程呢？呵呵。

####Reduce过程： 

1．Reduce会接收到不同map任务传来的数据，并且每个map传来的数据都是有序的。如果reduce端接受的数据量相当小，则直接存储在内存中（缓冲区大小由mapred.job.shuffle.input.buffer.percent属性控制，表示用作此用途的堆空间的百分比），如果数据量超过了该缓冲区大小的一定比例（由mapred.job.shuffle.merge.percent决定），则对数据合并后溢写到磁盘中。

2．随着溢写文件的增多，后台线程会将它们合并成一个更大的有序的文件，这样做是为了给后面的合并节省时间。其实不管在map端还是reduce端，MapReduce都是反复地执行排序，合并操作，现在终于明白了有些人为什么会说：排序是hadoop的灵魂。

3．合并的过程中会产生许多的中间文件（写入磁盘了），但MapReduce会让写入磁盘的数据尽可能地少，并且最后一次合并的结果并没有写入磁盘，而是直接输入到reduce函数。

##Hadoop 相关生态

###Hadoop Streaming
帮助开发者使用其他语言来发起 MapReduce 任务；实际就是把其他语言转换成 Java
###Pig、Hive
属于 High level interface，这俩都是基于 MapReduce ），帮助用户轻松地操作数据
###Azkaban
任务调度器，帮助用户组织管理一个负责的任务流程（Eventflow，包含一系列相互依赖的任务）
###Zookeeper
高性能分布式应用协调服务，以**Fast Paxos**算法为基础实现。提供关键信息的维持、同步、一致化（如一些分布式环境中重要的配置信息）
###Mahout
Hadoop 机器学习

#Spark
提供比MapReduce更丰富的特性，大量的适用于分布式计算的operation；重点是提供memory cache(号称加速读取100倍)、更直接的多语言支持(native python、scala、R)、以及 interactive shell。

##架构
三层结构，从底层到上层如下，Driver Program 也可以直接链接到 Worker Node;

- Worker Node: 和 MapReduce 的 Mapper、Reducer 节点区分不同，Spark 框架下的所有计算资源统称为 Worker Node，节点上运行的 Spark Executor Java Virtual Machine 是其核心，完成计算工作。这里也有 Move Computation to Data 的过程，Spark 会尽可能地分配任务到数据已经ready到本地的节点上。另外，使用 Java 或者 Scala 的话，代码可以直接跑在这个 Virtual Machine 上，PySpark 的话，会有 python 的进程起来，再 connect 到 Virtual Machine 上面。

- Cluster Manager: 对于节点管理（控制节点资源，管理节点状态等等），Spark supports YARN 和 Standalone (表示Spark自己进行节点管理)

- Driver Program: 就是跑在跳板机上的Client。拥有 Spark Context 维持上下文，和集群的Spark Instance保持连接。可以通过 spark-submit 提交一系列的jobs；也可以通过 spark-shell 来进行交互式的数据操作


##关键细节
- 其重要抽象是 Resilient Distributed Dataset (RDD)，数据资源将使用抽象的 RDD 对象来标识，然后我们可以对这个对象进行一系列操作。这个面向 RDD 的编程非常好的封装了底层的分布式运算细节！并且，其提供的 cache 是 Spark 性能优越的关键，对于一些需要重复使用数据集的算法（例如 Machine Learning 迭代训练），Spark 可以方便地把数据 cache 在内存里面来加速训练过程。

- Transformations 操作会保存在一个operator graph中，真正执行 Actions 的时候，会对这个 **DAG（Directed Acyclic Graph，非常重要~）**  进行根据各种依赖关系，进行整理、优化，产出最终直接序列（一些列包含map和reduce的阶段），通过 task scheduler 来最终执行；这整个过程中，中间过程的数据不写回磁盘的话，将会带来效率收益

##RDD
- Resilient Distributed Dataset，这是 Spark的数据存储方式，我们会把数据都对应到 RDD 对象，实际它上指向了集群上真正的数据集或者本地数据（HDFS、S3、HBase、Json等等）
- RDD 是不可修改的，只能通过 Transform 处理为其他的 RDD（创建了一个新的）
- 核心在于 Resilient，弹性地应对分布式计算中的 Failure

RDD 上的操作分为 Transformations 和 Actions，下述进行详细介绍：

### 自定义函数
分布式处理环境中，自定义函数操作的 associative 以及 commutative 是关键，这直接影响系统是否能够高效地优化分布式处理流程；例如 groupByKey 和 reduceByKey 适用完全不同的场景，要从需求上去考量是否有必要适用高成本方案。

###Transformations
这类方法会创造一个新的 RDD 对象，例如 map；并且他们都是 lazy 的，不会立即执行，只会在有 Actions 要求返回结果时才执行；可以用 persist 或者 cache 将 RDD 持久化在内存，也可以持久化在硬盘（当然，cache 还是要等到 Actions 执行了才生效）不 cache 住的话，可能在每次执行 Actions 时都重复计算

- flatmap & map: RDD 中的每个elem一般为一行源数据，map对应的是一对一执行function的过程，只会把一行数据映射到一个结果；所以我们有时候需要使用flatmap，先把原始的一行解开为多个数据，例如 WordCount 的 case，我们希望把一行句子映射会一系列单词，这个时候就需要用flatmap完成map映射（可能是1对多），并把所有结果都flatten到一块，当做新的element看待

- groupByKey：开销很大，需要把所有相关数据都shuffle到一起（key统一），再把所有value打包存在这个key下面。可以接受各种自定义处理函数
- reduceByKey：比groupByKey轻很多，可以各个机器上先reduce缩减计算规模，然后再shuffle合并计算结果并reduce，因为每个key只产出一个值且不需要数据放在一块计算。要求自定义函数 associative 且 commutative

###Actions
把数据做处理，返回给用户（driver program）最终结果
- reduce：这个是action，所以与reduceByKey不同


# Spark SQL
比较类似 pandas 的 dataframe；DataFrame（老版本叫作SchemaRDD） 和 普通的 RDD 的最大区别是，DataFrame 保持了schema信息，描述structured data。会有 selecting, filtering, aggregating 等等常见的结构化数据操作

### registerTmpTable("xxx")
Spark 提供这样一个函数，可以把 Dataframe 注册为一个虚拟的表（表名即“xxx”），然后支持sql查询~
eg. df.registerTempTable("people")
    sqlContext.sql("select age, count(*) from people group by age").show()







