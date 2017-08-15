#STL（Standard Template Library）
####vector
- 一块连续的存储，随机查、改效率很高；删、增效率低（需要受影响的所有元素移动）
- 每次存储超过限制，将会重新申请两倍大的存储空间，并转移已有数据；事先可以通过 reserve 预申请足量空间，免得多次倒腾（例如你用for循环一直插入数据）

####string
字符串

####list
双向链表存储，高效的删、增开头、结尾的数据；但是随机访问效率低下

####deque
介于vector和list之间的页式存储；即提供高效的

####set、multiset、map、multimap
基于 红黑树（RBTree）实现的 map 结构

####hash_set、hash_multiset、hash_map、hash_multimap
非标准的STL容器，基于 hash 实现的~ 我的 X-code 里面并没有~

####stack、queue
非标准的STL容器，而是在 deque 的基础上封装的 adapter；提供标准的 栈、队列 结构

####priority queue 优先队列
非标准的STL容器；数据 push 入列enqueue后，优先级高的先 pop 出列dequeue；其实就是 binary heap 基于 vector 的一个实现

####sort、stable_sort、partial_sort
各类排序算法，参数为容器上开头、结尾的迭代器，还可以传入自定义的比较函数：  
```
bool less_(const myclass & m1, const myclass & m2) {
        return m1.second < m2.second;
}

sort(vect.begin(), vect.end(), less_);
```
####仿函数（functor）
C++中这一技术广泛被STL采用，又称为函数对象，实际上是定义一个类作为函数，重载了其运算符'operator()'，使其调用时如同执行函数一样，但是这个方法比函数指针高效。

#构造与析构（Constructor & Destructor）
- 构造函数；对象构建的时候调用，默认构造函数的话什么都不做
- 赋值构造函数；对象使用 等号 赋值时调用，默认赋值构造函数为浅拷贝
```
    Point & operator=(const Point& a) {
        cout<<"Assignment Constructor"<<endl;
        this->x = a.x, this->y = a.y;
        return *this;
    }
```
- 复制构造函数；对象复制（Object b = B(a)）时调用，默认复制构造函数为浅拷贝
```
    Point(const Point& a)
    {
        cout<<"Copy Constructor"<<endl;
        this->x = a.x, this->y = a.y;
    }
```
- 析构函数；对象销毁时调用

#堆的分配，new、delete 以及 malloc（memory allocate）、free
- **malloc与free是C++/C语言的标准库函数，new/delete是C++的运算符；它们都可用于申请动态内存和释放内存**
- 对于非内部数据类型的对象而言，光用maloc/free无法满足动态对象的要求。对象在创建的同时要自动执行构造函数，对象在消亡之前要自动执行析构函数。由于malloc/free是库函数而不是运算符，不在编译器控制权限之内，不能够把执行构造函数和析构函数的任务强加于malloc/free
- 因此C++语言需要一个能完成动态内存分配和初始化工作的运算符new，以一个能完成清理与释放内存工作的运算符delete。注意new/delete不是库函数
- C++程序经常要调用C函数，而C程序只能用malloc/free管理动态内存
- new可以认为是malloc加构造函数的执行。new出来的指针是直接带类型信息的。而malloc返回的都是void指针

#sizeof  
**这是一个很多坑的函数...充分展示出C/C++ 中数组和指针的差异...**  
- sizeof 对数组求取的是 **数组的所占字节大小**，所以数组在进行元素数量计算时应使用：sizeof(array) / sizeof(array[0])，这里是讲一维数组  
- sizeof 对指针求取的是 **指针本身所占字节大小**，所以在 32位 平台上任何类型指针都是 4，64位 平台上任何类型指针都是 8  
``` 
    //这是 32位 平台下的栗子
    
    char a[] = "hello world";
    char *p  = a;
    cout<< sizeof(a) << endl;   // 12字节（这里 strlen 结果是 11字节；sizeof 会计算上 '\0'）
    cout<< sizeof(p) << endl;   // 4字节

    void Func(char a[100])
    {
        cout<< sizeof(a) << endl;   // 4字节而不是100字节（形参的数组退化为指针）
    }
```

#memset
这个是完全根据字节来操作赋值的；一般是复制全 0，用于初始化数组。所以对 整数数组 memset 1 的话，得到不是一组整数 1，而是 $2^0+2^8+2^{16}+2^{24}$（因为 int 对应的4个字节都被赋值为 1）

#virtual
用于定义虚函数的关键字；virtual 关键字必须定义在基类的相关函数上（子类相关函数可以不加 virtual），之后通过基类的引用调用相关函数时，系统会动态查找出继承树上最新的一个实现（调用时动态绑定），表现出多态。其实，Java 中的话，所有函数都是虚函数...  
另外，函数可以被定义为没有实现的**纯虚函数**：
```
virtual void f() = 0;
```
此时，这个类就变成了抽象类，也相当于 Java 的 abstract 类，无法被实例化；这个类的继承者如果不实现相关 纯虚函数，也将是一个 abstract 类。如果一个基类全部函数都是纯虚函数，其实也就等效于 Java 的 interface 了。  
虚函数在子类中实现时，可以配合保留字 override 确保正确，效果和 Java 中一致。










