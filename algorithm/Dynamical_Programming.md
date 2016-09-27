#The Charm of Dynamical Programming 动态规划之魅
动态规划有如脑筋急转弯一样，让看似复杂的问题，突然间极其简洁。这种将问题等价、拆解的艺术，让编程顿时富有魅力！本文主要从最优子结构的角度，阐述几个经典的动态规划问题。

##Stock Exchange

##Backpack Problem

##Coin Exchange

##Matrix Multiplication

##Longest Increasing Subsequence 最长递增子序列
###问题：
设数字序列为 $array \in R^{\ n\times 1}$，这 n 个数字呈乱序排列，求整个 array 的最长递增子序列的长度   
###拆分：
$x_n$ 为最终目标，子问题 $x_i\ (1<=i<=n)$ 标识选取序列 $array$ 前 $i$ 个元素时，对应的最长递增子序列的长度  
###最优子结构：  
$x_i = max\{x_{j}+1, 1\},\ \  1 <= j < i\ \ \&\ \ array_j <= array_i$  
这里要搜索当前位置前的所有子问题，如果有子问题对应的数字小于当前位置的数字，那么这个子问题结果 +1 就有可能构成当前位置的最优解；当然，如果当前位置的数字小于之前的所有数字，那么当前位置的最优解只能是 1

##Longest Common String
  
##Maximum Sum of Subarray
这一问题非常经典！



