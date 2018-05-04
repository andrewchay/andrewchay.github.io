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

# Lecture 4 Model Free Prediction
This lecture talks about how to estimate the value function for an unknown MDP.

## Monte-Carlo Method
MC learns directly from complete episodes. It does not require the prior information of MDP. By law of large numbers, the MC mean will converge to the true value. All episodes must terminate in order to compute the mean.

At each time t, Monte-Carlo method calculate G(t) = R(t + 1) + gamma * R(t + 2) + ... + gamma ^ (T - 1) * R(T). Then v_pi(s) at state s can be calculated not using expectation but Monte-Carlo simulation. V(S(t)) = V(S(t)) + alpha * (G(t) - V(S(t))).


## Temporal-Difference Method
TD learns from incomplete episodes by bootstrapping. TD updates a guess towards a guess. When compared to MC method, TD(0) method updates V(S(t)) by replacing G(t) by R(t + 1) + gamma * V(S(t + 1)). This is a one-step approximation of G(t) when the full episode is not known. Hence TD can work in non-terminating environments.

When comparing TD against MC, TD introduces some bias but reduces variance by a lot. The reason is TD depends on one random action, where as MC depends on many random actions and rewards.

## TD(lambda)
### Forward View
TD(0) method utilizes one step look forward. It could be extended to multiple steps look forward naturally. TD(lambda) is a weighted average of these look forwards. Specifically, define G^(n)(t) = R(t + 1) + gamma * R(t + 2) + ... + gamma ^ (n - 1) * R(t + n) + gamma ^ n * V(S(t + n)), and G^lambda(t) = (1 - lambda) sum_{n = 1} ^ inf lambda ^ (n - 1) G^(n)(t). V(S(t)) = V(S(t)) + alpha * (G^lambda(t) - V(S(t))).

### Backward View
Forward view provides theory, whereas backward view provides mechanism. E_0(s) = 0, E_t(s) = gamma * lambda * E_{t - 1}(s) + I(S(t) = s). This is called the eligibility trace. V(S(t)) = V(S(t)) * alpha * delta(t) * E_t(s), where delta(t) = R(t + 1) + gamma * V(S(t + 1)) - V(S(t)).

Backward view and forward view of TD(lambda) is equivalent when the update is offline. In practice, Exact Online update is equivalent to forward view.


# Model-Free Control
## On-Policy Learning
Learn about policy pi from experience sampled from pi.

Greedy policy improvement over V(s) requires model of MDP. Greedy policy improvement over Q(s, a) is model-free.

e-Greedy exploration method is as follows:
pi(a|s) = e / m + 1 - e if a is the greedy policy, or e / m if otherwise. This ensures the continual exploration. 

Greedy in the Limit with Infinite Exploration (GLIE) property: All state-action pairs are explored infinitely many times as the number of episodes goes to infinity and the limit of policy converges on a greedy policy. This property ensures all state-action pairs are explored and the convergence of the greedy policy. e-greedy is GLIE if e_k = 1 / k where k is the index for episodes.

Since TD control covers MC control as a special case, only TD control is noted in this notes. And the algorithm is called state-action-reward-state-action (SARSA), which is used to update Q(S, A) as Q(S, A) = Q(S, A) + alpha * (R + gamma * Q(S', A') - Q(S, A)). The algorithm goes as follows:

1. Initialize Q(s, a) for all s, a arbitrarily and Q(terminal-state, .) = 0.
2. Repeat (for each episode):
3.   E(s, a) = 0 for all s, a
4.   Initialize S
5.   Choose A from S using policy derived from Q (ie, e-greedy).
6.   Repeat (for each step of episode):
7.     Take action A, observe R, S'
8.     Choose A' from S' using policy derived from Q (ie, e-greedy)
9.     delta = R + gamma * Q(S', A') - Q(S, A)
10.    E(S, A) = E(S, A) + 1
11.    For all s, a:
12.       Q(s, a) = Q(s, a) + alpha * delta * E((s, a))
13.       E(s, a) = gamma * lambda * E(s, a)
14.      S = S', A = A'
15.  until S is terminal

Sarsa converges  to the optimal action as long as alpha(t) satisfies sum of alpha(t) = infinity and sum of alpha(t) ^ 2 < infinity, and GLIE sequence of policies pi_t(a|s)

## Off-Policy Learning
Off-policy learning learns from other policy than the target policy pi. This is important because we can use it to learn from human agents, reuse experience generated from old policies, learn about optimal policy while following exploratory policy, or learn about multiple policy while following one policy. 

### Q-Learning
Consider the Q(s, a), behavior policy mu and alternative policy pi. The behavior policy mu is e-greedy w.r.t. Q(s, a). The target policy pi is greedy w.r.t. Q(s, a). Then the Q-learning target simplies to R(t + 1) + gamma * max_{a'} Q(S(t + 1), a').  

1. Initialize Q(s, a) for all s, a arbitrarily and Q(terminal-state, .) = 0.
2. Repeat (for each episode):
3.   Initialize S
4.   Repeat (for each step of episode):
5.     Choose A from S using policy derived from Q (ie, e-greedy).
6.     Take action A, observe R, S'
7.    Q(S, A) = Q(S, A) + alpha * (R + gamma * max_{a'}Q(S', a') - Q(S, A))
8.    S = S'
9.   until S is terminal