Graph Algorithm

# 常用工具
- Python networkx
- Spark GraphX，这个后续做一些调研



# 一些基本概念
- 分为 directed graph 有向图、undirected graph 无向图
- 有向图的 vertex 有 in-degree 入度、out-degree 出度
- edge 可能有权值
- 常用数据结构 邻接矩阵 adjacency matrix 表示图
- 连通分量（子图）称为 connected_components

- 聚类系数 (clustering coefficient)


可以计算整个图中，所有顶点的聚类系数的平均数，来衡量整个图的聚集情况。python networkx中称为 average_clustering



# 图的一些热门算法

## Graph Analysis 
- PageRank
- Shortest Path
- Graph Coloring

## Community Detection
- Triangle Counting
- K-core Decomposition
- K-Truss

## Collaborative Filtering
It is a graph algorithm... also a matrix factorization algorithm...
- Alternating Least Squares
- Stochastic Gradient Descent
- Tensor Factorization



