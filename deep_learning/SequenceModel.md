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

Actually, it is a basic component to support OCR, Speech Regnition, Machine Translation and etc. Because it judges which answer is most possible.

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


## 3. Word Embedding

### A very key insight is transfer learning
The word embedding itself can be trained with very large corpus. Thus it absorb so much information and may known more than millions words.

After that, we can uses embedding in daily scence like sentiment recognition with tiny training set(the cost of label is much higher~) to get very good generalization.














