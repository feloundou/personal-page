---
title: The Components of an Action
date: 2021-02-16T18:00:07.668Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
Reinforcement learning literature involves learning to pursue actions that provide sufficient enough rewards (or minimize the agent's cost). As we have previously seen, encouraging continuous actions defined as an "environment step" can be tricky because of the **credit assignment problem,** wherein the learning function must attribute credit for rewards or costs to some actions taken along a trajectory**.** 

Human decision-making, however, routinely extends far into time. Choosing to go to Cancun or Aspen for winter break involves a long series of actions, including purchasing flights, securing housing, buying or renting downhill skiing or water skiing gear, taking time off from work, etc... Within each of these decision-making "blocks", you encounter mini-steps, i.e. to purchase a flight, you must choose whether to peruse Kayak or Google Flights, enter the correct dates, decide among your choices on dimensions of price, duration of flights, number of layovers, etc... 



Examples such as the above help illustrate at what different scales actions can be abstracted to involve a long sequence of small steps over time. However, the classical building block of reinforcement learning, the MDP, has no notion of temporal continuity between actions. Conventionally, they define the reward at time t, to be a direct result of the action taken at time t-1, with no longer scale time dependency.

This is where the concept of the **sMDP** comes in. Under the **semi-Markov Decision Process** framework, actions take variable amounts of time and are intended to model actions that extend beyond one time-step.