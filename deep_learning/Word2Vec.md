# WordEmbedding
*WordEmbedding 是词嵌入的方法论 & Word2Vec 是google的一种建模及实现手段。*
## 引入
对于 one-hot encoding，不同的 word 之间完全是隔绝的，没有任何的关联；每一个 word 都需要充分的样本数据来让模型学习。
然而 WordEmbedding 构造了一个 featurized representation 的嵌入空间，将 word 隐藏的语义关联在了一块。

## The Key Impact of WordEmbedding
最大的优势就是我们可以使用超大量 corpus（语料）学习出 WordEmbedding，这里面包含了对各个 word 的深入理解，尤其是不同 word 之间的关联。之后在特定问题建模，复用这个已经获得的 embedding，可以大幅降低特定问题的习难度，增强 generalization（特定问题的训练数据可能非常有限，但是通过已有的 embedding 信息可以大幅推广开）。

另一方面，数据的输入维度会降低：100k维度的one-hot向量可能降为300维度的dense。当然，其实这里的输入信息量是显然提高了~（由单个整数id，变为300个浮点数）


实际上的insight是 Transfer Learning，典型套路如下：

- learn word embedding from very large corpus (billions, even hundred billions). Besides, you can just donwload pre-trained embeddings online (such as Embeddings from Tencent-AI-lab).
- transfer embedding to new task with small train set (maybe only thousands)
- (optional) fine tune embedding if you have large train set of your own task


## Embedding's fancy feature： Analogy
There are lots of fancy case:

- Man - Woman + King = Queen
- Man vs. Woman as Boy vs. Girl
- Ottawa vs. Canada as Nairobi vs. Kenya
- Big vs. Bigger as Tall vs. Taller
- Yen vs. Japan as Ruble vs. Russia

Embedding is a latent semantic space (feature-like representations) that catches the relationships between words. Thus we can do some analogical reasoning. 

(but need very very large corpus to learn these things~)


## Embedding Matrix
WordEmbedding is actually a dense matrix such as 300(rows)*100k(columns). And there are 2 challenges:

- There are so many parameters to learn! (a dense vector for each word) Thus it is a tough task that requires huge corpus data and computation resource.
- This embedding may cost too much memory in serving time.

Another skill for embedding is that: 
we do not use multiplication operation to extract representation vector from embedding matrix, which spend too many useless multiplications(0*x=0) and not effient. "lookup" operation is used instead to extract specific column from embedding matrix directly. In keras, we often use build-in EmbeddingLayer to do that.


## Learning WordEmbedding

### Idea from Language Model
[Bengio et. al., 2003, A neural probabilistic language model]

It is kind of a language model that using the context to predict the target word, but the aim is just get proper embeddings from parameters of neural network.
The structure of Nerual Network is simple: Concat of embeddings of context -> 1 hidden layer -> softmax to loss. In a word, building a mapping (a probability distribution) from context words to targe word. 

To learn the embedding, the context can be defined as different form:
(example here: I want a glass of orange juice to go along with my cereal)

- 4 words on left & right (8 embeddings to predict target word): a glass of orange & to go along with -> juice
- just 1 left word: orange -> juice
- nearby 1 word (skip-gram model): glass -> juice
(If you really want to build a language model, using only 4 words on left makes more sense. But the context above is practical for learning embedding.)

Besides, softmax meets a big problem because vocabulary size may over 100k (just number of classification), it is hard to directly use softmax loss.


### Word2Vec, Skip-grams Model
[Mikolow et. al., 2013. Efficient estimation of word representations in vector space.]
This one is simpler. The key difference is training set.

#### Building Trainset context->target
Randomly pick pairs from corpus including a context word and another target word within some window(such as -5,+5 window). Thus the trainset only contains pairs that one word to another word.

#### Sampling of context word:
The frequently used but meaningless words like a/the/of/to/etc. may dominate training samples. Thus some sampling heuristic about word frequncy or stop-word-list matters.

#### Model Structure
The embedding of a context word -> softmax. Details of model showed below:
$$p(target|context)=\frac{e^{\theta_{target} emb_{context}}}{\sum_{j=1}^{10,000}e^{\theta_j emb_{context}}}$$

The embedding of context word is used to predict the probabilty of target word. $\theta$ is the parameter of softmax (output layer), $emb$ is the parameter of embedding.

#### Problem of softmax
The denominator of softmax sum all vocabulary and is impossible to compute (even harder if you have larger vocabulary size). Thus hierarchical softmax or negative sampling is used to solve this problem.

#### Hierarchical softmax
Hierarchicalk softmax classifier builds a tree (not complete tree~) for classification that cost log|vocabulariy_size|. Besides, Huffman (consider frequncy and path-length) tree is commonly used to make the method even quicker.

Key Idea: Any target word has and only has one specific path in the tree from root to leaf. The leaf node represent the target word, and the path node is a binary-classification node and own its parameters as $\theta$.
Thus softmax problem for n words become a series binary-classification for log|n| path nodes, which dramatically save computation cost. And the final predict result is just multiplication of log|n| binary-classification results rather than a complete softmax. And the loss function is the sum of all loss of bineary-classification.

Here is a very clear blog for that: https://blog.csdn.net/itplus/article/details/37969979


### Negative sampling
[Mikolow et. al., 2013. Distributed representation of words and phrases and their compositionality.]
This one is even simpler. The key difference is define a binary-classification problem rather than multi-classification. Clearly, there is no computation cost problem for softmax~

#### Model
For a sample(context -> target as positive sample), k negative samples are also sampled (context -> random words) to build the binary-classification training set. 
The number k is a hyper-parameter here and there are some heuristic:

- k=5-20 for small dataset
- k=2-5 for larger dataset

A example of training set is below (k=4):

| context | target | label |
| ------:| ------: | ------: |
| orange | juice | 1 |
| orange | king | 0 |
| orange | head | 0 |
| orange | at | 0 |
| orange | car | 0 |

Model Structure is just:
$$p(y=1|c,t)=sigmoid(\theta_t^Te_c)$$


#### Sample Skill
How to select negative sample is the key.

- only considering global frequence is bad. (lots of stop words like a/the/of/and/etc.)
- just uniform for all words ($\frac{1}{|v|}$) is also bad. (donot consider the distribution words)

Thus Mikolow found a balance to define sample rate for any word $w_i$ as below:
$$p(w_i)=\frac{freq(w_i)^{3/4}}{\sum_{j=1}^{10,000}freq(w_j)^{3/4}}$$

The hyper-parameter $3/4$ sort of suppress the high-frequency words. But for me, I also want to drop all stop words.


### Glove algorithm
[Pennington et. al., 2014. GloVe: Global vectors for word representation]

It is a really different algorithm that using linear regression to learn wordembeddings~

#### Label
$x_{ij}$ defined as #times of word i appears in the context of word j.
$x_{ij}$ and $x_{ji}$ equals or not just depends on the definition of context->target words. It can be a symmetric relationship.

Thus $x_{ij}$ actually represents how close these two words are.

#### Model
The loss function is blow (just a square loss):
$$min_{\theta,e}\sum_i^{10,000}\sum_j^{10,000} f(x_{ij})(\theta_i^Te_j+b_i+b_j-logx_{ij})$$

##### weighting
$f(x_{ij})$ here called weighting term or weight factor, and adjust 3 cases:

- drops the samples that $x_{ij}=0$
- controls stop words, and may suppresses high-frequency words
- lifts up some low-frequency words

It is the heuristic part that controls the balance, which just like the negative sampling skill.

##### final embedding
$\theta$ and $e$ are symmetric for this algorithm, thus the final embeddings defined as below:
$$e^{(final)}=\frac{\theta+e}{2}$$

It is kind of a regularization.





