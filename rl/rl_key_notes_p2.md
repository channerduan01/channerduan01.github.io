## Policy Gradient
The algorithms mentioned above are value-based methods, which evaludates the value of actions first and then make decision (also call implicit policy). However there is another family of reinforcement learning algorigthms called **Policy Gradient** are policy-based.

Policy Gradient directly predicts the actions(distribution) without evaluation of the value of actions like value-based model Q-learning/DQN. Thus it can deal with infinite/continous action that DQN cann't.

The essence of Policy Gradient is it encourages(gives higher probability to choose) the actions result in good reward while discourages the actions result in bad reward.

[paper of Policy Gradient](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

[slide if Policy Gradient](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf)

### 1. REINFORCE (basic version of Policy Gradient)
First, there are 2 kinds of modeling:

- Softmax Policy: solves discrete action from limited action set.
- Gaussian Policy: solves continuous action. Commonly implemented by predicting the mean and std of Gaussian distribution.

Here is a basic example, the NN model struture of it is quite similar to DQN, but there are some key differences about training.

```
function REINFORCE
	Initalise theta arbitrarily
	for each episode {s1,a1,r1,s2,a2,r2,...} ~ PI_theta do
		for t = 1 to T do
			update theta
		end for
	return theta
end function
```
The "update theta" is below (round update):
$$\theta = \theta +\alpha \nabla_{\theta}(log\pi_{\theta}(s_t,a_t)v_t)\\\ v_t=normalize(r_t)\ on\ all\ {r_1,r_2,...}$$

Actually, the loss function is:

$$loss = -log\pi_{\theta}(s_t,a_t)v_t$$

- $\theta$ are the parameters of NN model
- $\alpha$ is just the learning rate
- $\pi_{\theta}(s_t,a_t)$ is the model/policy's predict probability $\in(0,1)$ of the policy to take action $a_t$ for state $s_t$
- $v_t$ is the utility value and which is also the weight of the sample(can be positive or negative). It is actually the normalized reward calculated for one episode(a whole round of a game).

#### 1.1 model output
**For Softmax Policy**: it is still a NN model like the one of DQN for discrete actions, and the only difference is one more **Softmax** layer on output. Thus the output of this model $\pi_{\theta}(s_t,a_t)$ is a discrete probability distribution of actions.

**For Gaussian Policy**: the NN-model output is 2 parts:

- the mean of Gaussian distribution; one-dim value using tanh activation, and rescaled to the bound size.
- the std of Gaussian distribution; one-dim value using softplus(ensure positive), and added a base value to ensure some kind of exploration.

#### 1.2 loss(gradient ascend in this format)
$log\pi_{\theta}(s_t,a_t)v_t$ encourages the action with good reward:

- lower predict probability of the action gets larger gradient
- higher weight($v_t$) of the action(for that state) gets larger gradient

It is actually cross-entropy with weight, but all the positive label are the actions that policy chosed. The key is the weight $v_t$ contains reward infomation.

(A trick: the entropy of distribution can be added to encourage exploration)

#### 1.3 weight $v_t$
Go through this round/episode and compute the samples weights.

**First**, evaluating the value of each pair of (state,action):
$$q(s_t,a_t)=\sum^{T-t}_{k=0}\gamma^{k}r_{t+k}$$

The key is the hyperparameter **$\gamma$** combines short term and long term rewards, which is also used in Q-learning.

**Then** producing $v_t$ by normalizing the $q(s_t,a_t)$ that the mean is 0 and the std is 1. Thus, good pair gets positive weight while bad pair gets negative weight.

For calculating $v_t$, this algorithm have to do round update. It usually makes training much slower.

#### 1.4 round/episode update
Because of $v_t$, it has to wait a round/episode terminates and cache all samples to compute weights $v_t$ then do training. However, Temporal-Difference update (Q-learning/DQN uses) is more efficient.

Besides, this algorithm does not use experience replay. Thus all samples are dropped after learn.

#### 1.5 infer
Though DQN uses episilon-greedy, REINFORCE does not use it. The predict result(a probability distribution) naturally introduces some exploration. There are 2 styles:

- Softmax Policy selects the action with highest probability of state $s$.
- Gaussian Policy samples the action(a coutinuous value) from Gaussian Distribution of corresponding state $s$, and uses clip to ensure the bounds.

### 2. Actor Critic
It is the combination of Policy Gradient(Actor) and Value-based method(Critic). The running process of Actor Critic for each step includes these 3 parts:

- Actor makes decision on predict probability.
- Critic scores Actor's action.
- Actor updates based on Critic's score results.

The key difference to REINFORCE is using **Critic** to evaluate the weight/score $TD_{err}$.

#### 2.1 Actor
Actor is the same as the one in REIFORCE. It also directly makes the desicion of action, but it is trained after every step using the $TD_{err}$ from Critic (instead of $v_t$ in REINFORCE).

#### 2.2 Critic*
Critic is a NN-model has one-dim value, which evaluates the $TD_{err}$ using current state $s$, next state $s'$ and the reward $r$:

$$TD_{err}(s)=r+\gamma V(s')-V(s)$$

- $V(s')$ is the NN-model's predict value of next state $s'$
- $\gamma$ is the hyperparameter shrinks the future reward(commonly seen in rl~)
- $r+\gamma V(s')$ is the learning target/label of Critic, just like the target/real Q in Q-learning
- $V(s)$ is the the NN-model's predict value of current state $s$, just like the predict Q in Q-learning

**Then**, $TD_{err}(s)$ is used to train both Actor's NN-Model and Critic's NN-Model:

- Actor uses $TD_{err}(s)$ as weight in policy gradient. The positive $TD_{err}$ means the policy leads to a better state $s'$(or the state $s$ is actually better than it thought), and the policy should be encouraged, and vice versa.

- Critic directly uses $loss=TD_{err}(s)^2$ to train. It it the same as DQN.

#### 2.3 Compare to REIFORCE
- Advantage: it makes policy gradient update for every step(Termporal-Difference update)
- Disadvantage: very hard to train and converge.

### 3. DDPG (Deep Deterministic Policy Gradient)
An improvement model proposed by Google DeepMind **for modeling continuous action**, which combining DQN with Actor Critic to make it converge effeciently.

#### 3.1 Key differences
- Deep; Deeper NN-model.
- Deterministic; No sampling from probability distribution for output infer. But just deterministicly chooses the best action even for continuous action.
- Policy Gradient; A different definition of the gradient. It is really direct and elegant.

#### 3.2 Modeling
- **the predict Actor** is $\theta_{\mu}$, it directly predicts the value of action itself. **It is deterministic.**
- **the real Actor** is $\theta_{\mu '}$, hold and sync the parameters from predict one
- **the predict Critic** is $\theta_{Q}$, it predicts the value of action $a$ for a state $s$, just like the DQN. In this case, the continuous actions are part of the input of NN-model.
- **the real Critic** is $\theta_{Q'}$, hold and sync the parameters from predict one


The update/learn of **the predict Actor** is:
$$\nabla_{\theta_{\mu}}J \approx \frac{1}{N}\sum_{i} \nabla_{a} Q(s,a|\theta_{Q})|_{s=s_i,a=\mu(s_i)}\nabla_{\theta_{\mu}}\mu(s|\theta_{\mu})|_{s=s_i}$$

It is actually a chain rule that directly deliveries the gradient(information) from Critic(Q-function) to the parameters of Actor NN-model. It encourages the Actor to get better Critic. 

The update/learn of **the predict Critic** is:
$$y_i=r_i+\gamma Q'(s_{i+1},a_{i+1}=\mu'(s_{i+1}|\theta_{\mu'})|\theta_{Q'})\\\nabla_{\theta_{Q}}J = \frac{1}{N}\sum_i(y_i-Q(s_i,a_i|\theta_{Q}))^2$$
It is really similar to standard DQN, the only difference is the action $a_{i+1}$ of next state $s_{i+1}$ is determined by **the real Actor** NN-Model. Then the loss of Critic is just the MSE between real and predict Q-values.

#### 3.3 Training Loop
The key is also Q-value. And the train data is the same as DQN: 

tuple of (current state $s$, action $a$, reward $r$, next state $s'$).

- Evaluating **the real Q**($y_i$ above) using both the **the real Actor**(take $s'$ as input and output $a'$) and **the real Critic**(take $s', a'$ as input and output $Q'$).
- Updating **the predict Critic**(take $s, a$ as input and output $Q$) to approximate **the real Q**($y_i$)
- Updating **the predict Actor**(take $s$ as input and output $a$) using **the predict Q**($Q$) that encourages Actor to make action for higher Q-value(using $\nabla_{\theta_{\mu}}J$ above)
- Synchronizing the parameters from predict NN-model to real NN-model after some iterations

#### 3.4 Other details
It also uses **Experience replay** in DQN to cache experience (current state $s$, action $a$, reward $r$, next state $s'$), then sample train date from memory and do batch training, which breaks the dependency of experience and makes Neural Network learn and converge more efficent.

Moreover, for the final infer/output of Actor, it is really the value of actions. There is nothing about probability.

### 4. A3C (Asynchronous Advantage Actor-Critic)
It is also a advanced version of Actor Critic, which proposed by Google DeepMind. It is on-policy.

#### Asynchronous
The key is Asynchronous, which means distributed/concurrency here. The A3C builds one global node as to save global parameters for Actor-Critic, then builds many concurrent worker nodes to run Actor-Critic on their local-envs and push the gradient and pull the parameters to sync to global node.

It is kind of similar to **Parameter Server**, and there are some key points:

- work node owns a totally separate local environment to play with(get feedback).
- work node owns all the parameters, but doesn't change them until pull from global node.
- work play in local env for some time(maybe 1 episode or 20 steps), then sync. It is async.


### 5. DPPO (Distributed Proximal Policy Optimization)
Proposed by OpenAI and Google DeepMind.

The key is to control the policy's updating scale by the KL penalty: 
$$\lambda KL[\pi_{old},\pi_{new}]$$

It is also asynchronous/distributed.


## Some Documents

伯克利大学人工智能公开课：http://ai.berkeley.edu/lecture_videos.html

https://zhuanlan.zhihu.com/p/25239682

https://blog.csdn.net/jinzhuojun/article/details/72851548

https://zhuanlan.zhihu.com/p/52169807

https://www.zhihu.com/question/310681355/answer/619422276



2015年，DeepMind的Volodymyr Mnih等研究员在《自然》杂志上发表论文Human-level control through deep reinforcement learning[1]，该论文提出了一个结合深度学习（DL）技术和强化学习（RL）思想的模型Deep Q-Network(DQN)，在Atari游戏平台上展示出超越人类水平的表现。自此以后，结合DL与RL的深度强化学习（Deep Reinforcement Learning, DRL）迅速成为人工智能界的焦点。
