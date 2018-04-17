This is the notes taken from David Silver's Reinforcement Learning course
# Lecture 1 Introduction

## Three components of reinforcement learning
1. Observation from the ***environment*** at time t O(t)
2. Reward from the ***environment*** at time t R(t)
3. Action taken from the ***agent*** at time t A(t)

## History is the sequence of the observations, rewards, and actions
H(t) = O(1), R(1), A(1), ..., O(t), R(t)

## State is a function of history that is used to determine what happens next
S(t) = f(H(t))

## Three components of agents
1. ***Policy*** is the mapping from state to action
2. ***Value*** is the expected future reward
3. ***Model*** is the understanding of what the environment going to do next

Depending on the existence of the three components, different RL agents can be constructed.

## Balance between exploration and exploitation
Exploration finds more information about the environment. Exploitation maximizes the reward given the known information. This is like the balance between model interpretability and predictability. 

# Lecture 2 Markov Decision Process
## Markov Reward Process
The Markov reward process consists of (S, P, R, gamma). Here S is the state space. P is the transition matrix from S to S. R(s) is the expected reward at time t + 1 when the state at time t is s. gamma is the discount factor.
Define return G(t) = R(t + 1) + gamma * R(t + 2) + ...
Value function v(s) = E(G(t)|S(t) = s), which is the expected long term return.

## Bellman Equation for MRP
The matrix form of Bellman equation is (1 - gamma * P) * v = R

## Markov Decision Process
Markov decision process adds the action A into Markov reward process. 
A policy pi is a distribution over actions given states, pi(a|s) = P(A(t) = a | S(t) = s).

## Value Function
### State Value Function
v_pi(s) = E_pi(G(t) | S(t) = s)
### Action Value Function
q_pi(s, a) = E_pi(G(t) | S(t) = s, A(t) = a)

Hence the Bellman expectation equation for MDP is 
v_pi(s) = E_pi(R(t + 1) + gamma * v_pi(S(t + 1)|S(t) = s). 
q_pi(s, a) = E_pi(R(t + 1) + gamma * q_pi(S(t + 1), A(t + 1)|S(t) = s, A(t) = a).

v_pi(s) and q_pi(s, a) satisfies the two equations:
1. v_pi(s) = \sum_{a in A} pi(a|s) q_pi(s, a);
2. q_pi(s, a) = R_s^a + gamma * \sum_{s' in S}P_ss'^a * v_pi(s')

Recursively, v_pi = R^pi + $\gamma$ * P^pi * $v_pi$.

The optimal state-value function v_ * (s) = max_pi v_pi(s). q_ * (s, a) = max_pi q_pi(s, a). The optimal policy is a policy that is better than or equal to all other policies in terms of v_pi(s) >= v_pi'(s) for any s. Then
v_ * (s) = max_a q_ * (s, a), q_ * (s, a) = R^a_s + gamma * \sum_{s' in S} P^a_ss' v_ * (s').

# Lecture 3 Planning by Dynamic Programming

