---
title: Learning from OpenAI Experts
subtitle: Imitation Learning
date: 2020-12-05T03:38:14.296Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
An emergent trend of my posts so far has been my attempt to link my progress through curriculum study to concepts in artificial intelligence. The end of my learning is no different. Still, I consider the fact I tie my own personal learning to the learning process that autonomous agents undergo to be **a feature, not a bug.**

Having studied convolutions, attention, actor-critic and generative methods, I came full circle to reinforcement learning. The basic premise of reinforcement learning is that an agent aims to maximize, the (weighted) sum of rewards she receives from a given **environment**. The environment is such that agents find themselves in an initial state, take an action, find themselves in a different state, then choose what next action to take. The transition from state to state is governed by **Markov decision processes**. These provide a mathematical framework for transitions from state to state, such that the outcome of a decision is partly due to a past action taken and partly random (much like life). Therefore, by trial and error, the autonomous agent learns what are the best actions to take and when. 

There are myriad algorithms to teach agents how to select the best (most rewarding/least costly) action and when. The choice of algorithm can depend on the situation the agent finds himself in. If the environment is fully observable and the state-action space is not , you can simply predict state transition, 

To motivate my research question, I will use a Confucius quote:

>  “By three methods we may learn wisdom: First, by reflection, which is noblest; Second, by imitation, which is easiest; Third, by experience, which is the bitterest.

In the artificial intelligence context, we could per