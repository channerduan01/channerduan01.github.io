## Policy Gradient
The algorithms mentioned above are value-based methods, which evaludates the value of actions first and then make decision (also call implicit policy). However there is another family of reinforcement learning algorigthms called **Policy Gradient** are policy-based.

Policy Gradient directly predicts the actions(distribution) without evaluation of the value of actions like value-based model Q-learning/DQN. Thus it can deal with infinite/continous action that DQN cann't.

The essence of Policy Gradient is it encourages(gives higher probability to choose) the actions result in good reward while discourages the actions result in bad reward.

[paper of Policy Gradient](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

[slide if Policy Gradient](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf)

### REINFORCE (basic version of Policy Gradient)
This is very simple algorithm for policy gradient. Actually the NN model struture of it is quite similar to DQN, but there are some key differences.

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

#### 1. model output
It is still a NN model like the one of DQN for discrete actions, and the only difference is one more softmax layer on output. Thus the output of this model $\pi_{\theta}(s_t,a_t)$ is a discrete probability distribution of actions.

#### 2. loss(gradient ascend in this format)
$log\pi_{\theta}(s_t,a_t)v_t$ encourages the action with good reward:

- lower predict probability of the action gets larger gradient
- higher weight($v_t$) of the action(for that state) gets larger gradient

It is actually cross-entropy with weight, but all the positive label are the actions that policy chosed. The key is the weight $v_t$ contains reward infomation.

#### 3. weight $v_t$
Go through this round/episode and compute the samples weights.

**First**, evaluating the value of each pair of (state,action):
$$q(s_t,a_t)=\sum^{T-t}_{k=0}\gamma^{k}r_{t+k}$$

The key is the hyperparameter **$\gamma$** combines short term and long term rewards, which is also used in Q-learning.

**Then** producing $v_t$ by normalizing the $q(s_t,a_t)$ that the mean is 0 and the std is 1. Thus, good pair gets positive weight while bad pair gets negative weight.

For calculating $v_t$, this algorithm have to do round update. It usually makes training much slower.

#### 4. round/episode update
Because of $v_t$, it has to wait a round/episode terminates and cache all samples to compute weights $v_t$ then do training. However, Temporal-Difference update (Q-learning/DQN uses) is faster.

Besidees, this algorithm does not use experience replay. Thus all samples are dropped after learn.

#### 5. infer
Though DQN uses episilon-greedy, REINFORCE does not use it. The predict result(a probability distribution) naturally introduces some exploration.

## Actor Critic
It is the combination of Policy Gradient(Actor) and Value-based method(Critic, eg. DQN). It makes policy grradient update for every step(Termporal-Difference update)

- Actor makes decision on predict probability
- Critic scores Actor's action
- Actor updates based on Critic's score results


### DDPG (Deep Deterministic Policy Gradient)
Combining DQN with Actor Critic to make it converge better.



