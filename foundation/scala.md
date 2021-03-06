Scala学习笔记

We must consider that Scala:
is completely object oriented: every value is an object
is strongly typed: every value must have a type

# Overview
Scala, short for Scalable Language, is a hybrid functional programming language.
Good Website for Scala: https://www.tutorialspoint.com/scala/index.htm


Scala is a modern multi-paradigm programming language designed to express common programming patterns in a concise, elegant, and type-safe way. 



val line = sc.textFile("/home/hdp-guanggao/antispam/minerva/test/hadoop/input")
line.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_+_).collect().foreach(println)


# 一些关键点

- 不推荐使用return
- No Null
- 一切都是对象，包括Null（Null是为了兼容java，scala标准中，应该是None）


- Option/Some/None pattern

- Scala 中的 Any 相当于 Java 中的 Object，是所有对象的超类

- 类型查看与强制类型转换：classOf、isInstanceOf、asInstanceOf：这三个关键字后面加入[]，放入要查询的类型名称。eg. classOf[String]:
 总而言之，我们把classOf[T]看成Java里的T.class, obj.isInstanceOf[T]看成 obj instanceof T, obj.asInstanceOf[T]看成(T)obj就对了。scala为我们提供了语法糖，但也免不了类型擦除问题的影响。

- 获取对象类型："123".getClass.getSimpleName  


- List、Array、Tuple
    Same: 长度都是固定的，不可变长
    Difference:
        Array 中的元素值可变，List和Tuple中的元素值不可变
        Array一般是先确定长度，后赋值，而List和Tuple在声明的时候就需要赋值
        Array取单个元素的效率很高，而List读取单个元素的效率是O(n)
        当使用混合类型时，Array和List会将元素类型转化为Any类型,而Tuple则保留每一个元素的初始类型
        访问方式不同，Array和List的下标从0开始，且使用小括号,而Tuple的下标从1开始，切使用点加下划线的方式访问，如：arrayTest(0), listTest(0); Tuple访问: tupleTest._1

- 操作符号默认函数，是一个很大的坑
  eg.  val tt = ArrayBuffer[(Int, Int)]()
       tt += ((1,1))
  上述必须用两个括号，否则会出问题。第一个括号默认为函数'+='的括号，第二个括号才是Tuple的定义括号~




# 函数式编程与 Scala
## 解释性编程语言：

- 修改变量
- 可以赋值
- 包括很多控制语句，例如：if-then-else、loops、break、continue、return

## 纯函数式编程语言：

- 没有任何可变变量
- 没有循环（for, while）
- 使用递归函数控制
