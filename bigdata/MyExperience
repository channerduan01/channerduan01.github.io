大数据经验

# join 问题
join 是传统数据库应用中非常常用的操作，在大数据领域，Map-Reduce 以及 Hive、Spark 中也经常应用。
join 操作非常地消耗性能（时间+空间），必须谨慎对待；对于不同的数据场景，我们有不同的具体实现方式优化，以下展开简要介绍。

## 经典的 Hash Join
这个方法的核心是把数据的 Key 映射到一个 Hashtable 来关联俩个表，是经典的单机 join 算法；整个过程经历三步：

- 确定Build Table以及Probe Table：这个概念比较重要，Build Table使用join key构建Hash Table，而Probe Table使用join key进行探测，探测成功就可以join在一起。通常情况下，小表会作为Build Table，大表作为Probe Table。此事例中item为Build Table，order为Probe Table。
- 构建Hash Table：依次读取Build Table（item）的数据，对于每一行数据根据join key（item.id）进行hash，hash到对应的Bucket，生成hash table中的一条记录。数据缓存在内存中，如果内存放不下需要dump到外存。
- 探测：再依次扫描Probe Table（order）的数据，使用相同的hash函数映射Hash Table中的记录，映射成功之后再检查join条件（item.id = order.i_id），如果匹配成功就可以将两者join在一起。

相对的是 Mysql 使用的  Nested-Loop Join，其性能较低~

## broadcast hash join
将其中一张小表广播分发到另一张大表所在的分区节点上，分别并发地与其上的分区记录进行hash join。broadcast适用于小表很小，可以直接广播的场景。
**这里有一个陷阱**：广播一个小表来join大表的场景中，必须是 LeftOuterJoin 广播右表 或者是 RightOuterJoin 广播左表，否则毫无意义；实践中一定要注意这个细节，当然Spark肯定是自动优化的~

## Shuffle Hash Join
一旦小表数据量较大，此时就不再适合进行广播分发。这种情况下，可以根据join key相同 必然分区相同的原理，将两张表分别按照join key进行重新组织分区，这样就可以将join分而治之，划分为很多小join，充分利用集群资源并行化。

## Sort-Merge Join
- shuffle阶段：将两张大表根据join key进行重新分区，两张表数据会分布到整个集群，以便分布式并行处理
- sort阶段：对单个分区节点的两表数据，分别进行排序
- merge阶段：对排好序的两张分区表数据执行join操作。join操作很简单，分别遍历两个有序序列，碰到相同join key就merge输出，否则取更小一边，

## 注意要点
Spark 默认就是采用这种 Sort-Merge Join 方式，包含 shuffle过程。这里有一个shuffle的通病：当数据的key严重偏斜（例如某个key对应的行数超过其他所有key对应行数之和），使用 shuffle 会造成非常大的性能障碍：一个数据量非常大的key会造成大量的shuffle，并且之后由单个节点处理join任务，可能造成这一节点的运行时间是其他节点的上千倍，严重影响分布式性能。

