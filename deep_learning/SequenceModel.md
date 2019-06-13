# Sequence Modeling of Deep Learning

**Several different types of Sequence Modeling (the mapping from x to y):**

- 1(even 0) to Seq: Music Genereation, Poems Generation
- Seq to 1: Sentiment Classification (Comment Analysis), Activity Recognition

- Seq to Seq (same length): Entity Regnition; predict $y^{t}$ for each $x^{t}$, predicting the type/property of each word within a sentence

- Seq to Seq (different length):
Machine Translation (Very Popular!), Speech recognition; Mapping a seq to another seq(may not the same length), which is a process called Encoder and Decoder. The model gets all information from input seq and encode to a State in some feature space (high dimensional), then decode this State to a different space (such as another language).

- Attention Based 

## 1. Recurrent Neural Networks

### 1. Intuition
It really looks like a State Machine~

- The State hold by rnn-cell itself.
- This machine is scanning input sequence from left to right(or right to left as bi-rnn) and changing its State.
- Current State depends on current input(one separate input from sequence) and Last State

### 2. Basic RNN-block:
Supposed $a^{t}$ is the state at time $t$, and input sequence is: $x^{t}, t \in [1,2...,T]$;

$$a^{t}=g(W_{aa}a^{t-1}+W_{ax}x^{t}+b_a)$$

$W_{aa}$ is sort of State transformation matrix, and $W_{ax}$ absorbs input, and activation function $g()$ is usually tanh/relu here.

- the formula can be simplified as $a^{t}=g(W_{a}[a^{t-1},x^{t}]+b_a)$
- $a^0=\vec{0}$ is used for $t=1$
- the same $W$ and $b$ is shared for all time $t$, and the size of matrix $W$(length of vector $b$) depends on how many nodes you defined for this RNN-cell
-  the output of each state is defined as: $y^{t}=g(W_ya^{t}+b_y)$


### 3. More sophisticated RNN-block

There are a lot more advanced ways to define the State Machine.

#### 3.1 Background: Vanishing Gradient Problem for RNN
Basic RNN is hard to capture the long-range dependency. There is a good example:

"The cat, which already ate ... , was full"

"The cats, which is blue ... , were full"

**was/were** is decided by the noun **cat/cats**, which is really far. Basic RNN may settle 2 or 3 word range well but cannot solve this case because of gradient vanishing. Thus we need LSTM/GRU.

By the way, for the opposite problem Exploding Gradient (relative easy problem), "gradient clipping" skill can be used to solve it.


#### 3.2 GRU(Gated Recurrent Unit)
Gated Recurrent Unit, which is common used to make long-range connections. Actually, the structure of GRU is based on lots of experiment rather than mathematics, and it is robust in the most cases.

**Key:**

The key is that using gate to hold useful state and really memorize important information/context.

Two Gates is introduced to mange the memory of cells:

- $\Gamma_{r}$ controls the weight of old/last state to infludence candidate state
- $\Gamma_{u}$ controls using new candidate state or holding old/last state

**Detailed Model:**

- for each time, two gates are built:
$$\Gamma_{u}=sigmoid(W_u[c^{t-1},x_t]+b_u)$$
$$\Gamma_{r}=sigmoid(W_r[c^{t-1},x_t]+b_r)$$
(the value of sigmoid is between 0 and 1, it intuitively controls open or close.)

- then, a candidate state is generated:
$$\hat{c^t}=tanh(W_c[\Gamma_{r}c^{t-1},x_t]+b_c)$$

- after that, the new state is built:
$$c^t=\Gamma_{u}\hat{c^t}+(1-\Gamma_{u})c^{t-1}$$

- finally, the output of each state is defined as: 
$$a^{t}=c^{t},\ \ y^{t}=g(W_ya^{t}+b_y)$$

And the most important of all, the state $c^{t}$ is actually a vector, and $\Gamma_{u}$, $\Gamma_{r}$ do have the same size with it. Thus there can be 100 gates(vector!) at the same time to control and memorize different stuff. The power of GRU is massive. 

#### 3.3 LSTM(Long & Short Term Memory)
Long & Short Term Memory unit is even more powerful than GRU (also more complicated~).

**Key Difference to GRU:**

Comparing to GRU that owns two gates to control the impact of old state and constitute of new state, LSTM owns three gates:

- update gate $\Gamma_{u}$ controls whether or not using new inner state(momery) 
- forget gate $\Gamma_{f}$ controls whether or not using old inner state(momery) 
- output gate $\Gamma_{o}$ controls whether or not output the inner state(memory) to real output of the cell

("update gate + forget gate" is more flexible than only "update gate" in GRU)

and separate states:

- $c^{t}$ is inner state(memory) of cell
- $a^{t}$ is the outer state(output) of cell

**Detailed Model:**

- for each time, three gates are built:
$$\Gamma_{u}=sigmoid(W_u[c^{t-1},x_t]+b_u)$$
$$\Gamma_{f}=sigmoid(W_f[c^{t-1},x_t]+b_f)$$
$$\Gamma_{o}=sigmoid(W_o[c^{t-1},x_t]+b_o)$$

- then, a candidate state is generated (without ratio gate to fade the influence of last state):
$$\hat{c^t}=tanh(W_c[c^{t-1},x_t]+b_c)$$

- after that, the new state is built by two gates (thus it is possible to totally keep last state and totally add new state ):
$$c^t=\Gamma_{u}\hat{c^t}+\Gamma_{f}c^{t-1}$$

- next, the output gate is used to influence the outer state of the cell:
$$a^{t}=\Gamma_{o}tanh(c^{t})$$

- finally, the output of each state is defined as: 
$$y^{t}=g(W_ya^{t}+b_y)$$


### 4. BRNN(Bidirectional RNN)
- RNN absorbs the input sequence from left to right and capture the connections with earlier inputs.
- BRNN uses two RNN including left to right and right to left, which capture the connections for both directions.

The drawback of BRNN is that it requires the entire sequence of data before making any prediction.For Realtime Speech Recognition system, standard BRNN needs to wait users stop to get the whole sentence (bad experience~).

### 5. Deep RNN
RNN block also can be stacked. It is very hard to train thus usually within 3 layers. And it is common to insert simple full connected layers between RNN results and output layer.


## 2. RNN for Language Model
**The core is: predict the probability of a sentence.**

Actually, it is a basic component to support OCR, Speech Recognition, Machine Translation and etc. Because it judges which answer is most possible.

We commonly use word-level model, but character-level model is also possible in the future.
(character-level is a lot harder to capture dependency because the sequence is longer, and it requires more computation)

### 1. Presentation of word:
- Vocabulary (could be 100 thousand common words, it is actually a huge dictionary)
- Specific Token (such as \<EOS\> as End Of Sentence, and \<UNK\> as UNKnow word)

### 2. Modeling
- Input: the whole sequence of the sentence (supposed with length L). A trick is that the first input(t=1) of RNN is "None", which predicts the prior of the first word of any sentence globally
- Output: for each time, RNN predict the probability of $P(y_t|y_{t-1}...y_{t=1})$; for the first time, output is just $P(y_{t=1})$
- Language Model: Multiply all results of prediciton and the final result is probability of existence of the whole sentence

### 3. Sample Sequences from Model
Once a RNN Language Model is trained, novel samples could easily be generated from the model.
For the first word, just random sample the $P(y_{t=1})$. Then, the last word $P(y_{t-1})$ can be random sampled and choosed one as input to predict $P(y_t)$. It is easy to generate a whole sentence. It is really cool that machine knows how to write absolutely novel sentence by itself~


## 3. Sequence to Sequence Model
**just based on RNN for language model.**

[Sutskever et al., 2014. Sequence to sequence learning with neural networks]

[Cho et al., 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation]

### 1. Difference

Language Model: $P(y_1,y_2,...,y_{T_y})$, targets the existence probability of sentences. And it can even be used to generate novel sequence.

Sequence to Sequence (eg. Machine Translation): $P(y_1,y_2,...,y_{T_y}|x_1,x_2,...,x_{T_x})$, it is actually *conditional language model*, which encodes the $x$ sequence into a dense vector (representation of the whole $x$), then decodes to find the most likely $y$ for this dense vector condition.


They are really similar and there are just 2 additions of Machine Translation:

- Language Model uses all zero $a_0$ as the first input (thus first output is just prior), but Machine Translation uses the representation vector as $a_0$ which is the result encode network.
- Language Model may be used to sample and generate different novel sequence, but Machine Translation only cares the most likely translation $y$ using beam search. It just depends on usage scenario~


### 2. Beam Search

The target of this search algorithm is just below:

$$arg max_{y} P(y_1,y_2,...,y_{T_y}|x_1,x_2,...,x_{T_x})$$

Compare to other search algorithm:

- Exhaustive Search is impossible because the exponential possible of sentence ($10k^{10}$ for 10k vocabulary and 10 words sentence)
- Greedy Search is really limited that only consider one step furthers.

Beam Search is something between them, and it is not guaranteed to find exact maximum (compare to BFS/DFS). 
And notice: beam search is used to search the most possible result (serving), the whole model is already trained before beam search. It is just a search algorithm (strategy) and nothing about optimization or training.


#### Beam Width
For any search iterantion $t$:

$$arg max_{y_t} P(y_1,y_2,...,y_t|x)$$

$$P(y_1,y_2,...,y_t|x)=P(y_1,y_2,...,y_{t-1}|x)*P(y_{t}|y_1,y_2,...,y_{t-1},x)$$

Beam Search keeps the top $n$ results, which called *beam width*. Notice: different catch may have total different sequence (they do not share prefix such as $y_1,\ y_2$). The computation scale is fixed and it degenerates to Greedy Search when beam width = 1.

Beam Search with a width 3 to 10 always do much better than simple Greedy Search in Machine Translation scenario. The larger width gets more possible sequence, thus the value of it is a tradeoff between precision and computation cost. For production systems, width 10 is  enough; but for research width can be 1000. 

In practice, the nerual network is copied beam width times and concurrently run to make efficient parallize.


#### Maximum Log-Likelihood rather than original Likelihood
Actually, the search target conditional probability is below:

$$arg\ \max_y P(y_1,y_2,...,y_T|x)=\prod_{t=1}^TP(y_t|y_1,...,y_{t-1})=P(y_1|x)*P(y_2|y_1,x)*P(y_3|y_1,y_2x)*...*P(y_T|y_1,y_2,...,y_{T-1},x)$$

Thus directly multiplications of many small probability results in too small float value and cause underflow. A common solution is just taking log likelihood:

$$arg\ \max_y logP(y_1,y_2,...,y_T|x)=\sum_{t=1}^TlogP(y_t|y_1,...,y_{t-1})$$

Log function is strictly monotonically increasing which keeps the same max result for orginal probabiblity. And summation of log probability is numerically stable. (but may cause overflow~)

#### Length Normalization (get much better result)
There is length bias of the condicitional probability. The longer sequence unnaturally tends to have lower probability, because the probability of extra word definitely less than 1 (and log-value less than 0).

To deal with this bias, the search target normalized as below:

$$arg\ \max_y \frac{1}{T^{\alpha}}logP(y_1,y_2,...,y_T|x)$$

$\alpha$ is a hyper-parameter to control normalization strength:

- $\alpha=0$ is no normalization
- $\alpha=1$ is full length normalization
- $\alpha=0.7$ is a softer

#### Complete Workflow
For a Beam Width $B=3$, and a maximum search length $T=20$, beam search keeps top 3 results for each length(1 to 20) using log-likelihood. Then the length normalized score is calcuted for these 3*20 result and the most possible result is chosen. A sequence ends with token <EOS> maybe prefered.

#### Error Analysis on Beam Search
It helps to figure out the error is the problem of beam search or RNN encode-decode Model, and can be used to tuned hyper-parameters such as beam width and $\alpha$ of length normalization.

For $P(y|x)$ in machine translation, supposed $y^*$ is the best human translation (label), $\hat{y}$ is the most possible result found by beam search. $P(y^*|x)$ and $P(\hat{y}|x)$ can be calcuted using encode-decode model. When these 2 sequence not equal:

- $P(y^*|x)>P(\hat{y}|x)$: fault of beam search.
- $P(y^*|x)<P(\hat{y}|x)$: fault of RNN.  (maybe not proper length normalization)

You can go through the whole dataset and have statistics on which one cause more faults.

## Attention mechanism

The intuition of attention mechanism also comes from machine translation. For example, when we tranform a long English sentence to China, we would like to focus the begining part first rather than memory the whole sentence right away. It is just the difference between 'Sequence to Sequence' with 'Attention'.

### Common Model Structure
For machine translation problem, the model is consist of 3 parts: 

#### pre-attention Bi-LSTM
It absorts all information from input sequence with length $T_x$, and transform the sequence to cell states $a^{<1>},a^{<2>},...,a^{<T_x>}$. Moreover, each cell state is a concatenation of forward-direction activation and backward-direction activation to capture more sequence information.

It is a separate procedure(to process raw data) that just running one time before the rest 2 steps.

#### attention mechanism (The key!)
helping post LSTM pay more attention on some parts of the results of pre LSTM. 
To compute each input for the post LSTM $t \in [1,T_y]$, there are several steps:

- the last state $s^{t-1}$ of post LSTM is needed with the output results from pre-attention Bi-LSTM $a^{<1>},a^{<2>},...,a^{<T_x>}$

- the attention varables $a^{t,t'}$ is calculated by $softmax([[s^{t-1},a^{<1>}],[s^{t-1},a^{<2>}],...,[s^{t-1},a^{<T_x>}]]W_{attention})$, which is a distribution of the importance $a^{<t,1>},a^{<t,2>},...,a^{<t,T_x>}$ for the results of pre LSTM.

- multiplying the attention varables $a^{<t,t'>}$(it is actually the importance weights) with raw results from pre-attention Bi-LSTM and sum up to produce attention results $context^{<t>}=\sum_{t'=1}^{T_x}a^{<t,t'>}a^{<t'>}$

The key point is that: $s^{t-1}$ influences the attention distribution, then influences output of attention layer indirectly.

#### post-attention LSTM
It is a common LSTM that goes forward from left to right. For each time step, the inputs include the last state $s^{t-1}$ and attention result $context^{<t>}$. It really works with attention mechanism that gives the last state $s^{t-1}$ and gets $context^{<t>}$ back.




