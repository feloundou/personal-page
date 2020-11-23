---
title: Learning with Rewards
date: 2020-11-23T06:25:30.894Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
Reinforcement learning is the subset of machine learning in which an agent exists within an environment and looks to maximize some kind of reward. The agent takes an action, which alters the environment in some way, observes the reward associated with the environmental change. The agent observes the new state and chooses her next action, repeating the process until a terminal state is reached. 

Q-Learning is a model-free approach to help the agent to recognize or learn the optimal action in every state visited by the system (also called the optimal policy) via trial and error.

* Trial and error can be implemented in a real-world system (commonly seen in robotics) or within a simulator (commonly seen in management science / industrial engineering).
* The algorithm works as follows:

  An agent chooses an action, obtains feedback for that action, and uses the feedback to update its database. This is sometimes called **Tabular Q-Learning.**

  * In its database, the agent keeps a so-called Q-factor for every state-action pair. When the feedback for selecting an action in a state is positive, the associated Q-factor’s value is increased, while if the feedback is negative, the value is decreased.
  * The feedback consists of the immediate revenue or reward plus the value of the next state. In short, the value of the next state is the discounted value of all future states that are reachable from that next state. 
  * Moreover, the value of any state is given by the maximum Q-factor in that state. Thus, if there are two actions in each state, the value of a state is the maximum of the two Q-factors for that state.
* When we have a **very large number of state-action pairs**, it is not feasible to store every Q-factor separately.

  * Then, it makes sense to store the Q-factors for a given action within a neural network. Thus, when a Q-factor is needed, it can be fetched from the neural network.
  * When a Q-factor is to be updated, the new Q-factor is used to update the neural network itself.
  * For any given action, Q(i, a) is a function of i, the state. Hence, we will call it a Q-function in what follows.
* For reinforcement learning, we need **incremental neural networks** rather than **batch updating** since every time the agent receives feedback, we obtain a new piece of data that must be used to update some neural network.
* Neurons are used for fitting linear forms, e.g., y = a + bi where i is the input (the state in our case). 
* Backpropagation is used when the Q-factor is non-linear in i, which is usually the case. (Algorithm was invented by Paul Werbos in 1975). Backprop is a universal function approximator, and ideally should fit any Q-function! #backpropagation