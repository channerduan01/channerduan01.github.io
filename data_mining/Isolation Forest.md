# Isolation Forest (iForest)
It is a state of art algorithm for Anomaly Detection. It is a fundamentally different model-based method that explicitly isolates anomalies instead of profiles normal points. It is an ensemble model that consists by iTrees.

## Basis
### Some Terms for Outlier Detection
- masking (an outlier is undetected as such)
- swamping (a nonoutlier is classified as an outlier)

### Assumption
Anomalies are **'few and different'**
They are minority and have attribute-values that are very different. Thus they tend to be isolated easily
### Advantages
- No distance function need
- Graceful handle missing value
- Could provide anomaly explainations
- Parameter Free
(https://www.youtube.com/watch?v=L1WWv_v   Bigmal)

### Disadvantages
- Not really Parameter Free...  
The number of tress and sub-sampling size are specified

## Structure - Isolation Tree (iTree)
- Proper Binary Tree (sometimes Strictly Binary Tree)
Let T be a node of isolation tree. Then T is either an external-node with no child or an internal-node with one test and exactly two children nodes. So the memory upper bound for iTree is **2n-1**. It is similar to BST (Binary Search Tree)
**The memory requirement is bounded and only grows linearly with n**
- Build the Tree
Given dataset X = {$x_1, x_2, x_3, ..., x_n$} in m-dimension, we recursively divide X by randomly selecting an attribute m and a split value p(between max and min) util either: (i)the tree reaches a height limit (ii)$|X|=1$ (iii)all data in X have the same values

## **Anomaly Score**
Path Length reveals the anomaly (remember our assumption about anomalies). The random partitioning produces noticeable shorter paths for abnormal samples!   
The score of a sample x is defined as below:  
$$s(x, n) = 2^{-\displaystyle\frac{E(h(x))}{c(n)}}$$  

- $n$ is the total number of our samples  
- $E(h(x))$ is the average path length of sample x in all the trees
- $c(n)$ is the average path length given $n$! So it is used for any samples like a truth from nature. Actually, it is calculated by harmonic number based on another paper ($c(n) = 2H(n-1)-(2(n-1)/n)$).  

Based on the equation above, the anomaly score is a value between (0,1), the bigger the more abnormal.

## Characteristic of Isolation Forest
### Sub-sampling size (basic requirement for isolation)
**It is kind of tricky but it is the key of this algorithm** Isolation method works best when the sampling size is kept small. It cannot efficiently isolate and find anomaly when the sampling size is too large. It alleviates the effects of swamping and masking.
- It helps iForest better isolate examples of anomalies
- Each isolation tree can be specialised as each sub-sample includes different set of anomalies or even no anomaly (feature-sampling may make further effect)
  
It is really important to find a appropriate sub-sampling size.

## Using Isolation Forest
First, the whole forest is built on the sub-sampling from original data. Then, the path lengths for all the data are measured and the anomaly scores are calculated.
### Generate the Forest
The sub-sampling size $\psi$(default $2^8$) and number of trees $t$(default 100) should be specified. **The time complexity of training is: $O(t\psi \log{\psi})$**   

The tree height limit is calculated by: $l=ceiling(\log_2{\psi})$, which is approximatedly the average tree height. We are only interested in data that have short path length thus we donot need to really grow the whole tree structure.  
Then, we build all the iTrees.

### Evaluate the Score
For each data point, we travel through all the iTree to calculate the average height $E(h(x))$, then get the anomaly score using the equation mentioned before. n is the data size and
**The time complexity of evaluating is: $O(nt\log{\psi})$**  

For the external nodes that did not totally expanded in training stage, the c(.) function mentioned before is used to calculate the expected path length based on the number of samples stayed in this node.






 