#Embedding
这本来是一个拓扑学的词。地图就是对于现实地理的embedding，现实地理地形的信息其实远远超过三维，但是地图通过颜色和等高线等来最大化表现现实的地理信息。  
##Word Embedding
是用固定的维度来最大化表现词的信息；这是一个普适的概念，或者说任务。为了区别one-hot的词向量，可翻译成词嵌入。Word2Vec 是 Google 实现 Word Embedding 的一种具体approach。  
其本质上是根据context来对词向量进行降维操作。
VSMs(Vector space models) represent (embed) words in a continuous vector space where semantically similar words are mapped to nearby points. 这一方面在 NLP 已经有很长的研究历史，不过所有的方法都依赖 Distributional Hypothesis, which states that words that apper in the same contexts share semantic meaning. 方法分为两大派系：count-based methods(eg. Latent Semantic Analysis) and predictive methods(eg.neural probabilistic language models)
###Word2vec
这是作为一个计算高效的 predictive model for learning word embeddings from raw text. 可以使用俩种不同的方式来做 CBOW(Continuous Bag-of-Words) model 或者 Skip-Gram model.




















