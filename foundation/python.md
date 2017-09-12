Python 积累

# 基础内容

## 字符串处理

### join 函数；通过连接符将一系列字符串连接为一个
	str.join(sequence)；str 标识分隔符，sequence 标识被连接的串
	eg. '.'.join(['1','2','3','4']) -> '1.2.3.4'

# 核心第三方库
## numpy 库
提供强大而灵活的矩阵数据结构 ndarray（上层接口为 array）

## linecache 库
行缓存，提供非常方便的文件读取控制

## networkx 库
图计算、图分析库

## pandas 库
表格容器，数据分析。基于 numpy 的 array（结构底层都是 array），提供了 Series 和 DataFrame 两个核心数据结构

### Series
类似于定长的有序字典，由 index（就是 key） 和 value 构成
- index			查看数据结构中的 keys，Index 数据结构
- index.values  具体的索引值 array
- values		查看数据结构中的 values，看到的是 array

- .to_frame 可以转为只有单列的 DataFrame

### DataFrame
DataFrame 类似是以 Series 组成的字典，具有行索引和列索引
- index			查看数据结构中的 keys，Index 数据结构
- index.values  具体的索引值 array
- values		查看数据结构中的 values，看到的是 array
- columns		查看列索引，就是各个列的命名

- loc[indics1,indices2] 提供表裁剪，indics1 标识提取的行，indics2 标识提取的列
	eg. df.loc[3:5,['A','B']]；注意，行一般是自然的整数索引，而列的话我们可能初始化了字符串为列命名，所以得指明具体命名

- drop_duplicates()		去重；直接调用会对所有列去重，可以用list指定列索引而只对部分列去重

- reset_index()			重置 index；默认将会把 index 插入表中称为新的 Column；设置 drop=True 可以丢掉原有 index

- groupby()		数据分组；
	参数 as_index(默认true) 决定是否将分组label设为结果的index
- merge()		连接查询（都用这个，没用 join 了）

### 条件筛选
可以使用DataFrame的.query()方法，或者直接使用boolean表达式筛选：


### 聚合函数
- size()
- count()		注意和 size() 的区分，这个函数不统计 NaN 值

### Tricks
- pd.isnull()		用于判断 NaN 值；类似于 np.isnan()；但是 np 的只能用于 np 内置类型
- 可以使用 value != value 来筛选出 NaN值，因为只有 NaN 不等于自己

### IO 操作
- 数据导入
	从CSV文件导入 DataFrame：
	eg. pandas.read_csv(dataFilePath,sep='\t',names=['a','b'...])，对
	sep 	标识数据分隔符
	names 	标识导入数据表各列的命名（不指定的话会使用数据源的第一列）

	从内置对象dict导入 Frame
	eg. pd.DataFrame({'cookie':list(cookieSet2)})
	
	从内置对象list导入 Series
	eg. pd.Series(list,dtype='str',name='cookie')
	name 	标识这一个序列的列命名

- 数据导出
	使用 DataFrame 或者 Series 的 to_csv() 方法直接导出
	eg.  df.to_csv(".\\s_ip",index=False,header=False)：index 标识是否写入索引列，默认会写入；还要注意指定字段分隔符 sep 

### 相关大坑
- dataframe 选取列的赋值问题，一下代码必须加上最后的 '.values'，否则赋值的会是 serie 的 index 数据
	raw_data.loc[indices_select, new_column_name] = data.loc[:, new_column_name].values

## sklearn
通用机器学习库

### 基本模型方法
- fit， 训练模型
- tranform， 数据转换，指的是特征抽取，特征转换，特征预处理等等
- dicision_function， 直接生成模型内部的评分（例如classifier用于最终决策的概率评价）
- predict， 生成预测
- score， 比对Groud Truth输出评分

### Pipeline
将机器学习的各个流程（特征提取+选择+模型训练）整合到一起，对于不同流程设定不同的名称，可以根据流程名称附加'__'来指定这个流程的参数名字，或者关闭一些流程。最终会按次序执行这一系列的Transform（特征转换）和一个最终fit（所以Pipeline只有最后一个step是estimator）
通常结合 GridSearch 使用

### FeatureUnion
相对于流水线 Pipeline，Union可以将一系列流程并行，来整合各个流程的结果，其本身又可以再嵌入到 Pipeline 中

### GridSearchCV （CV -> cross-validation）
网格搜索+交叉验证，用于模型调参；常配合 Pipeline 使用搜索大范围参数

### ParameterGrid
用于自动组织网格搜索相关参数，这里有一个点：for param_dict in list(ParameterGrid(parameters)) 输出的每一个搜索结果是一个dict，具体设置到pipeline里面需要解开，pipeline.set_params(**param_dict)

### externals.joblib dump & load。
模型的持续化以及载入

## matplotlib
画图 库
### plot

### scatter

### hist (histogram)
可以直接根据一维数据绘制直方图，快速呈现数值分布情况；横轴为从低到高的数值，数轴为各个部分数据数量；相当于自带数据整理，帮助我们快速可视化。可以再一张图上绘制多个分布进行比较，例如应用到 intra-distance 和 inter-distance 的比对中，又或是 TP、TN、FP、FN 的比对中 

##bar 更为底层的接口，直接传入具体的 x，y 来绘制出直方图，需要我们自己整理数据



