
# Markov decision process ( MDPs )

MDPs formally describe an environment for reinforcement learning. 
Where the environment is fully observable, and the current state fully characterises the process.

### Almost all RL problems can be formalised as MDPs:

- Optimal control primarily deals with continuous MDPs
- Partially observable problems can be converted into MDPs
- Bandits are MDPs with one state

It is really fundamental. There are 2 sides Agent and Environment with 3 connections Environment states $s \in S$, Agent actions $a \in A$ and Rewards $r \in R$.



### Markov Assumption/Property
The future is independent of the past given the present. (That is sort of philosophy~)
The current state is a sufficient statistic of the future.
$$P[S_{t+1}|S_t] = P[S_{t+1}|S_1,S_2,...,S_t]$$
$$P[s_{t+1},r_{t+1}|s_t,a_t] = P[s_{t+1},r_{t+1}|s_0,a_0,r_1,...,s_t,a_t]$$

Only needs to care the current state that captures all relevant information from history. Please throw away the past~ 


### State Transition Matrix
For a Markov state $s$ and successor state $s'$, the state transition probability is defined by:
$$P(S_{t+1}=s'|S_t=s)$$

State transition matirx $P$ defines transition probablities from all states $s$ to all successor states $s'$。
(index of row is 'from' and index of  )




# Some notes from Morvan


### Model-Free RL 不理解环境
- Q learning
- Sarsa
- Policy Gradients

### Model-Based RL 理解环境（需要为现实世界建模出一个虚拟世界）
其实就是多了多环境的建模，是的模型具有想象力，可以在虚拟世界中想象整个事态的发展（就像下棋的时候能够一直想象接下来的交手）
- Q learning
- Sarsa
- Policy Gradients


### Policy-Based RL 基于概率
- Policy Gradient

### Value-Based RL 基于价值
- Q learning
-Sarsa


### Monte-Carlo update 回合更新（整个游戏结束了才更新模型）
- Policy Gradient
- Monte-Carlo Learning

### Temporal-Difference update 单步更新（游戏中的每个步骤都迅速更新模型，更有效率）
- Q learning
- Sarsa
- Policy Gradient 

### On-Policy 在线学习（必须本人在场，边玩边学习）
- Sarsa

### Off-Policy 离线学习（看别人玩，根据别人的经验学习）
- Q learning
- Deep Q Network





# Some basic components and concepts about Reinforcement Learning

## Rewards
A reward is a scalar feedback signal
Indicates how well agent is doing at step t
Reward Hypothesis - All goals can be described by the maximization of expected cumulaive reward.

## Sequential Decision Making
Goal is to select actions to maximise total future reward
Actions may have long term consequences
Reward may be delayed
It may be better to sacrifice immediate reward to gain more long-term reward (you can't be greedy)

## Agent and Environment
take observation, make action, get reward

## History and State

The history is the sequence of observations, actions, rewards
$H_t=A_1,O_1,R_1,...,A_t,O_t,R_t$

What happens next depends on the history.

State is the information used to determine what happens next. Which replaces the history as a summary.
For normally, state is a function of the history: $$S_t = f(H_t)$$
- Environment State (We dont use it), it determines the observation/reward next. But this global state is private and not visible for agent. Agent only see very small part of the enviroment.
- Agent State (We use it), it contains information the agent got.

#### Information State
An information state contains all useful information from the history.
Definition of Markov: $P[S_{t+1}|S_t] = P[S_{t+1}|S_1,S_2,...,S_t]$
Once the state is known, the whole history can be thrown away.

#### Fully Observable Environment
agent directly observes environment state: $$O_t=S_t^{agent}=S_t^{environment}$$
agent state = environment state = information state
Formally, this is a Markov decision process (MDP)

#### Partial Observable Environment
agent state != environment state
Formally, this is a partially observable Markov decision process (POMDP)

#### Policy
A policy suggests which action to take, given a state.
Actually, the goal of reinforcement learning is to discover a good policy.
The optimal policy tells you the optimal action, given any state—but it may not provide the highest reward at the moment.


## Major Components of an RL Agent
Policy: agent's behaviour function, mapping state to action
Value function: how good is each state and/or action
Model: agent's representation of the environment









## 一些有用的链接

https://www.youtube.com/watch?v=fdY7dt3ijgY


https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf


https://zhuanlan.zhihu.com/p/32089487

https://github.com/junxiaosong/AlphaZero_Gomoku


https://www.youtube.com/watch?v=RtxI449ZjSc&feature=relmfu


https://skymind.ai/wiki/deep-reinforcement-learning


https://www.youtube.com/channel/UCP7jMXSY2xbc3KCAE0MHQ-A/videos


https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-A-DQN/


https://katefvision.github.io/





