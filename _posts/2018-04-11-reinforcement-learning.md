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