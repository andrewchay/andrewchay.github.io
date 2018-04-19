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

Recursively, v_pi = R^pi + gamma * P^pi * v_pi.

The optimal state-value function v_ * (s) = max_pi v_pi(s). q_ * (s, a) = max_pi q_pi(s, a). The optimal policy is a policy that is better than or equal to all other policies in terms of v_pi(s) >= v_pi'(s) for any s. Then
v_ * (s) = max_a q_ * (s, a), q_ * (s, a) = R^a_s + gamma * \sum_{s' in S} P^a_ss' v_ * (s').

# Lecture 3 Planning by Dynamic Programming

Dynamic programming is useful for problems when principles of optimality applies and optimal solution can be decomposed into subproblems.

For Prediction, we are given (S, A, P, R, gamma, pi), and need to output the value function v_pi. For control, we are given (S, A, P, R, gamma), and need to find the optimal value function v_ * and optimal policy pi_ * .

## Policy Evaluation
The problem is to evaluate a given policy pi in terms of its value v_pi. The way to evaluate is to apply Bellman expectation equation iteratively for a given pi. In matrix form, it is v^{k + 1} = R ^ pi + gamma * pi ^ pi * v ^ k.

## Policy Iteration
There are two steps to improve a policy:
1. Evaluate the policy pi: v_pi(s) = E[R(t+1) + gamma * R(t + 2) + ...|S(t) = s]
2. Improve the policy by acting greedily with respect to v_pi by finding a pi' that greedily maximizes v_pi over all pi.

This process always converges to pi_ * .

## Value Iteration

### Principle of Optimality
A policy pi(a|s) achieves the optimal value from state s, v_pi(s) = v_ * (s), if and only if for any states s' reachable from s, pi achieves the optimal value from state s', v_pi(s') = v_ * (s).

If we know the solution to subproblems v_ * (s'), then value iteration can be found by one-step lookahead v_ * (s) = max_{a in A} [R^a_s + gamma * sum_{s' in S} P^a_ss' v_ * (s')], and iteratively apply this rule.

## Extensions to Dynamic Programming
### Asynchronous Dynamic Programming

1. In-place dynamic programming: unlike synchronous value iteration, in which it stores two copies of value function, in-place value iteration only stores one copy of value function by keeping the updated values. Sweep all states from the largest remaining Bellman error.
2. Real-time dynamic programming updates onlyy states that are relevant to agent. 

### Contraction Mappting Theorem

Essentially Bellman expectation equation is a contraction mapping defined on state space and action space as long as gamma < 1. Hence it guarantees convergence to the unique best value.
2. 
