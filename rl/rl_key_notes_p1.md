## Reinforcement Learning

The core of Reinforcement Learning is Science of Desicion Making.

Reinforcement learning is an attempt to model a complex probability distribution of rewards in relation to a very large number of state-action pairs. This is one reason reinforcement learning is paired with, say, a Markov decision process, a method to sample from a complex distribution to infer its properties.

Recommended Textbook:
Algorithms for Reinforcement Learning, Szepesvari

[Recommended wiki]
(https://skymind.ai/wiki/deep-reinforcement-learning)

[Recommended Course on Youtube]
(https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT&index=2&t=0s)

#### There is a simple comparison:

| | Supervised learning | Reinforcement learning | Unsupervised learning |
| ------ | ------ | ------ | ------ |
| Target | Learning to approximate reference answers | Learning optimal strategy by trial and error | Learning underlying data structure |
| Label | Needs correct answers | Needs feedback on agent's own actions | No feedback required |
| Affect | Model does not affect the input data | Agent can affect it's own observations | Model does not affect the input data |


**The difference to Supervised Learning**:

- There is no supervisor, only a reward signal
- Feedback is delayed, not instantaneous
- Time really matters (sequential, non i.i.d data)
- Agent's actions affect the subsequent data it receives!


### Markov decision process
The problem format of RL.

### Bellman expectation equation
The heart of RL. It is about Dynamic Programing.


## Q-learning

### 1. Definition

Implementing the RL(reinforcement-learning) algorithm commonly requires 3 primary steps below:

- **infer (policy)**: select the best action($a$) from the current state($s$)
- **do** (it is more about the environment itself): perform the action
- **learn**: improve the understand of the world from the results (update Q-function for Q-learning algorithm)

Q-learning is an approach to solving reinforcement learning whereby you develop an algorithm to approximate the utility function (Q-function).
Once you got Q-function, the **policy** is really simple: 
$$\pi(s)=argmax_a\ Q(s,a)$$

Thus the core of Q-learning is how to define and learn the Q-function.

### 2. Utility Function (Q-function)

The utility of performing an action $a$ at a state $s$ is written as a function: $Q(s, a)$, which predicts the expected and the total rewards (the immediate reward + rewards gained later by following an optimal policy).

$$Q(s, a) = r(s, a) + \gamma \max Q(s', a')$$
It is a recursive function that meature the current utility upon immediate reward and the future rewards.

- $r(s, a)$ is immediate reward of action $a$ for state $s$.
- $\gamma$ is hyperparameter called discount factor. If $\gamma = 0$ then it is just greedy. And higher $\gamma$ considers more about the future. (pretty philosophy that we all put discount for future, otherwise the final ending of everything is death)

The **implementation** of Q-function is commonly a Q-table (rows for different states and columns for different actions).

### 3. Learning process
pseudo code of Q-learning is below:

```
Initialize Q(s,a) arbitrarily
with learning rate eta (may eta=0.1)
with discount factor gamma (may gamma=0.95)
Repeat (for each episode):
	Initialize s
	Repeat (for each step of episode):
		Choose a from s using policy derived from Q (eg. epsilon-greedy)
		Take action a, observe r, s'
		Learn Q(s,a) = Q(s,a) + eta[r+gamma max Q(s',a') - Q(s,a)]
		s = s'
	until s is terminal
```

The key update process is: 
$$Q(s,a) = Q(s,a) + \eta[r+\gamma max_{a'} Q(s',a') - Q(s,a)]$$

- $\eta$ is learning rate and $\gamma$ is discount factor
- **the predict Q** is " $Q(s,a)$ ", which predict value made by last Q-function (and we just about to optimize it)
- **the real Q** is " $r+\gamma max_{a'} Q(s',a')$ ", which is the learning target ("label")

**Notice**: the infer step("Choose a from s") commonly combined with epsilon-greedy(commonly 90% greedily choose best with 10% random) to random choose an action. It is the balance of explore & exploit.

## Sarsa
It is similar to Q-learning, which also uses Q-table to make decision. The only difference is about learning step.

### Learning process (Key difference from Q-learning)
```
Initialize Q(s,a) arbitrarily
Repeat (for each episode):
	Initialize s
	Choose a from s using policy derived from Q (eg. epsilon-greedy)
	Repeat (for each step of episode):
		Take action a, observe r, s'
		Choose a' from s' using policy derived from Q (eg. epsilon-greedy)
		Learn Q(s,a) = Q(s,a) + alpha[r + gamma Q(s',a') - Q(s,a)]
		s = s'
		a = a'
	until s is terminal
```

Sarsa does not use maximum to evaluate **real Q**, but directly chooses an action for the next state to determine **real Q**.

Thus the **learn** (for this step) and **do** (for next step) are consistent for Sarsa, it has to do the same thing in the next step thus it need on-policy (real time play and learn). For Q-learning, these are separate and it is off-policy (able to just learn from experiences).

- Sarsa is on-policy vs. Q-learning is off-policy
- Sarsa is conservative vs. Q-learning is brave

## Sarsa - Lambda
#### Learning process: think about the path
The key difference is consifering the whole path(all states from begin to the current), and update Q-values for all. It is usually more efficient.

There is a key hyperparameter Lambda(a float in $[0,1]$), which decay the weights of states in the path(the closer the more important). There some insight of different Lambda:

- Lambda $0$ is standard Sarsa (literally, it does not work for round update)
- Lambda $1$ gives the whole path the same update
- Lambda $(0,1)$ gives higher strength for the state closer to terminal



The rule of decay is: Eligibility-table $*=$ Lambda. And in practice, Eligibility-table is the same shape as Q-table, which uses to track the path and weights. 

The pseudo code is below:

```
Initialize Q(s,a) arbitrarily
Repeat (for each episode):
	Reset E(s,a) = 0, for all s, a
	Initialize s
	Choose a from s using policy derived from Q (eg. epsilon-greedy)
	Repeat (for each step of episode):
		Take action a, observe r, s'
		Choose a' from s' using policy derived from Q (eg. epsilon-greedy)
		error = r + gamma Q(s',a') - Q(s,a)
		E(s, :) = 0
		E(s, a) = 1
		For all s, a:
		   Learn Q(s,a) = Q(s,a) + alpha * error * E(s, a)
			Decay E(s, a) = lambda * E(s, a)
		s = s'
		a = a'
	until s is terminal
```

It can be changed from **Onestep-Update** to **Round-Update** easily by moving "Learn" out of the inner loop.


## DQN (Deep Q-Network)
#### 1. Combination of Q-learning with deep learning.

The traditional Q-learning can not hold unlimited states and actions in Q-table, and all the pairs of (state,action) is separated(as a cell in Q-table). Then Neural Network is introduced to replace Q-table. Which processes the raw features(using some skills like CNN/RNN) and learns underlying patterns with limited parameters and gets better generalization.

For Network of DQN model, there are 2 styles to replace Q-table:

- takes state and action as inputs(feature) and output Q-value
- takes state as input(feature) and output Q-values of all actions

The second one is commonly used. The last layer of that Nerual Network is linear(activate) layer, and the output number is just the **num of actions**. Besides, the input features are just feature about one state. It is a very simple and clean model.

### 2. The key skills of DQN
If you really want to train a DQN model, these 2 tricks below are inevitable.

#### Experience replay
Q-learning is off-policy and can learn from current experience, old experience and others experience.

For DQN, it maintains a repository of experience(train data), so the model can be trained on a large set of experience repeatedly. Besides, it randomly sample and replay(train) on each iteration, and the random sampling breaks the dependency of experience and makes Neural Network learn and converge more efficent.

The common format of train data is tuples as "(state, action, reward, next_state)":

- state: the last state
- action: the action agent took
- reward: the **instant** reward got from that action for that state
- next_state: the next state

Literally, there is no "label".

#### Fixed Q-targets
Fisrt, recalling the learning process of Q-learning below:

$$Q(s,a) = Q(s,a) + \eta[r+\gamma max_{a'} Q(s',a') - Q(s,a)]$$

- **the predict Q** is "$Q(s,a)$", which predict value made by last Q-function (and we just about to optimize it)
- **the real Q** is "$r+ \gamma  max_{a'} Q(s',a')$", which is the learning target ("label")

For the learning process of DQN, **the predict Q** and **the real Q** are computed by 2 different Networks(the same structure but different parameters $\theta,\theta'$), the latter one is just a much old version(so called fixed) of the former one, which makes the former Network much easier to learn and converge.

Besides, the learning problem becomes reducing the loss of $loss = \{r+\gamma max_{a'} Q(s',a';\theta') - Q(s,a;\theta)\}^2$ with optimizer, regularizer, hyperparameters like learning rate($\eta$) for **the predict Q** Network. (it requires some effort to implement the loss in practice)

And after some iterations(eg. 100 iters later), the parameters of latest **the predict Q** Network syncs to the fixed **real Q** Network. (In practice, the synchorinization may use some more complex strategy). 

Intuitively, **the real Q** is kind of label and used to train **the predict Q**. Thus the label is fixed at first to make sure the predict one can learn something and converge. After the predict one learned a little bit knowledge, the label replaced by it to be a slightly better label. And then gonna to train the predict even better. The model is becoming better and better circularly.

### 3. Learning process

pseudo code of Q-learning is below:

```
Initialize replay memory D to capacity N
Initialize action-value function(Neural Network) Q with random weights theta
Initialize target action-value function(same structure Neural Network) Q' with weights theta'=theta
Repeat (for each episode):
	Initialize s
	Repeat (for each step of episode):
		Choose a from s using policy derived from Q (eg. epsilon-greedy, and Q is NN here)
		Take action a, observe r, s'
		Store transition (s,a,r,s') in D
		Sample a random minibatch of transitions from D
		Perform gradient descent on (max_a'{Q'(s',a')}-Q(s,a))^2 with respect only to theta
		Every C steps reset(sync) theta'=theta(Q'=Q)
	until s is terminal
```

### 4. Unstable cost curve 
For reinforcement learning, the cost(on the training set) curve always go up and down lots of times through the training process, which is so different to the converged curve in supervise learning.

There are several clear reasons:

- The label is unstable. The target the model fits with is just the result of another model(old one), which changes after each parameters synchorization.

- The data may be unstable. While the model is evolving, it may take different actions with the same state and explore some unseen space, which may change the distribution of train data itself.

And a severe problem of reinforcement learning is it is hard to give an accurate offline evaluation. You need to run the policies online to see which one is better in practice. (Which is not serious for board game AI, such as Alpha Go runs a super scale search algorithm to play with AI as really fair evaluation)


### 5. Double DQN

It aims to solve the overestimating problem of DQN: the predict Q-value is higher than reality.

#### Overestimated Problem of DQN

**The real Q** (label) is: 
$$y_{target}^{(DQN)} = r+ \gamma  max_{a'} Q(s',a';\theta')$$

"$Q(s',a';\theta')$" is estimated by model which contains error, and that "max" operation maximizes the error. **The real Q** (label) itself is overestimated, which makes **the predict Q** "$Q(s,a;\theta)$" also overestimated after training.

#### Solution - Double DQN

Changes **the real Q** (label) as: 
$$y_{target}^{(Double-DQN)} = r+ \gamma Q(s', \arg \max_{a'}Q(s',a';\theta);\theta')$$

Which is now estimated by both Neural Network $\theta$ and $\theta'$. The former one computes the action with max Q, then the latter one directly computes **the real Q**.

In pratice, this method gives better(lower) predict Q-value.

### 6. DQN with Prioritised Replay

For the Experience replay of DQN, unifom random sample is just the baseline. The memory may have different priority/importance which gives different amount of informantion. Some prioritized replay strategies could dramatically speed up training process.

#### TD-error (Tempral-Difference-error)
Defines the priority/importance of a sample.
$$TD_{err} = |r+\gamma max_{a'} Q(s',a';\theta') - Q(s,a;\theta)|$$

It is introduced in these steps for DQN:

- New data/experience (collecting while training) gets max priority value(eg. 1) at first.
- Sample data using priority.
- Updated after each batch of training.
- The priority of data is also considered by the loss function. (with extra rescale/smooth step)

Moreover, the priority value needs some trick transform: 
$$p=\min(upper\_bound,TD_{err}+\epsilon)^{\alpha}$$

- "$\epsilon=0.01$" to avoid 0
- "$upper\_bound=1.$" in case of too large priority
- "$\alpha=0.4$" to smooth/rescale priority and gives low value more chance. Actually $\alpha=0$ means now priority while $\alpha=1$ means raw priority.


#### SumTree
In practice, the data struture SumTree is introduced to index and maintain memory/experience/training-set. Which makes sampling process much faster.

The space complexity is O(2n), with:

- n leaf node, priority of each data (all positive)
- n-1 branch nodes, priority sum of both children nodes

It is just array with size 2n-1 and the root node array(0) is the sum priority of whole data set.


For the sampling process, a random value of $[0, value_{root}]$ picked first, then go through the tree from the root to the leaf to find the corresponding pos of the random value. The time cost is just O(logn), and the pseudo code is below:

```
sample(node, value)
	if (node is leaf): return node
	gets left_node, right_node from node
	if (value < left_node.value): 
		return sample(left_node, value)
	else: 
		return sample(right_node, value-left_node.value)

sample(root_node, random_value)
```

In addition, the cost of update is also O(logn) that changes priorities from the leaf node to root.

[There is a good implementaion by Morvan](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py#L18-L86)


### 7. Dueling DQN
The only different is the structure of NN model, which defines a more complex Q-value function as below:

$$Q(s,a) = V(s,\theta)+A(s,a,\theta)$$


The common DQN model mentioned above(1~6) uses NN model to directly predict the Q-value for each action $a$ of some state $s$. However, dueling DQN assumes that Q-value is the sum of 2 parts:

- $V(s,\theta)$ is value of this state $s$ itself.
- $A(s,a,\theta)$ is advantage of some action $a$ for the $s$

And these 2 parts are outputs of the same NN model that share layers of feature extract. It makes the model more sensitive to different actions and converge faster.


#### Normalize $A(s,a,\theta)$ (an important trick)
It is really important and intuitive.
In practice, the NN model's output $A(s,a,\theta)$ may take over $V(s,\theta)$. And if $V(s,\theta)\approx 0$, it is not dueling. Thus the implementation of real Q-value function is:
$$Q(s,a) = V(s,\theta)+(A(s,a,\theta)-AVG(all\ A(s,a',\theta)))$$

It forces the $A()$ part to be really advantage/increment from this state.

### 8. Soft Replacement
It is a simple trick.

Just slightly changes(may by 1%) the target NN-model after every iteration of training rather than wait a long time for hard replacement/sync.