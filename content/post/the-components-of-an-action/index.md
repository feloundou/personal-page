---
title: The Makings of an Option
date: 2021-02-16T18:00:07.668Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
Reinforcement learning literature involves learning to pursue actions that provide sufficient enough rewards (or minimize the agent's cost). As we have previously seen, encouraging continuous actions defined as an "environment step" can be tricky because of the **credit assignment problem,** wherein the learning function must attribute credit for rewards or costs to some actions taken along a trajectory**.** 

Human decision-making, however, routinely extends far into time. Choosing to go to Aspen or Cancun for winter break involves a long series of actions, including purchasing flights, securing housing, buying or renting downhill skiing or water skiing gear, taking time off from work, etc... Within each of these decision-making "blocks", you encounter mini-steps, i.e. to purchase a flight, you must choose whether to peruse Kayak or Google Flights, enter the correct dates, decide among your choices on dimensions of price, duration of flights, number of layovers, etc... 

Examples such as the above help illustrate at what different scales actions can be abstracted to involve a long sequence of small steps over time. However, the classical building block of reinforcement learning, the MDP, has no notion of temporal continuity between actions. Conventionally, they define the reward at time t, to be a direct result of the action taken at time t-1, with no longer scale time dependency.

This is where the concept of the **sMDP** comes in. Under the **semi-Markov Decision Process** framework, actions take variable amounts of time and are intended to model actions that extend beyond one time-step. The limitation of SMDPs is that when they were first defined, there were no attempts to look into these actions with variable time horizons to examine or modify their sub-actions. Hence, the term **option** was devised to capture courses of actions with extended and variable time horizons. Note that actions, as defined in conventional MDPs, are a special case of options. Hence, one can model the choice between Aspen and Cancun as an option, and the choice between flights as either an option or a primitive action. The design and structure of this hierarchy can be difficult to define explicitly, e.g. choice between Aspen and Cancun can be modeled as an option or as a sub-option of the larger option of deciding to take a vacation. As RL researchers, we would rather have the agent discover them on their own. 

Options consist of 3 components: a **policy**, a **termination condition**, and an **initiation set.** If an option is taken, the policy guides the choices of actions until the termination condition is set. Once the option terminates, the agent has the opportunity to select another option, which is in the initiation set. 

These **options**, or temporally extended actions, does not define a natural way of learning what a good option might be. Automatically defining termination conditions and initiation sets presented a challenge. [One approach](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.2402&rep=rep1&type=pdf)  is to do this in two stages: in stage one, first the agent openly explores a static environment to gather statistics and choose potential sub-goals and initiation states. Statistics gathering in this case is related to credit assignment, that is, states that occur frequently on trajectories that successfully solve random tasks are more likely to be important. Therefore, we can see how the credit assignment problem is also applicable to the higher-level hierarchy of options. Agents may therefore develop **meta-control policies** that drive selection between options, and classical policies that selection primitive actions along the trajectory of the chosen option or meta-task.

Another approach, [DDO](https://arxiv.org/pdf/1703.08294.pdf), or Deep Discovery of Options, introduces an algorithm that uses an optimal agent to collect demonstrations at each hierarchical level. Note that this requires explicit definition of the hierarchies of each meta-option. At each level, DDO simultaneously

* infers option boundaries in demonstrations which segment trajectories into different control regimes
* infers the meta-control policy for selecting options as a mapping of segments to the option that likely generated them, and
* learns a control policy for each option.

Newer approaches, like [Diversity Is All You Need](https://arxiv.org/pdf/1802.06070.pdf) use an objective based on mutual information  to learn skills (used with nearly interchangeable meaning to options in this paper) without a reward. It works as follows: 

* The options/skills are required to be useful in the sense that each skill dictates the states that the agent visits. In other words, the state transition probabilities cannot be equal across states. This renders each skill distinguishable.
* Secondly, from the first requirement, you can use states visited to distinguish skills because actions that are not externally observable may still be important. For instance, an outside observer cannot tell how much force a robotic arm applies when grasping a cup if the cup does not move. 
* Lastly, encourage exploration by learning skills that act as randomly as possible. If skills do not explore states via actions that are far enough from states that other skills already visit, it will not be distinguish from the others. 

The DIAYN approach obviates the need to explicitly define skill hierarchies during learning, as demonstrated over some complex tasks, including locomotion.

Human decision making is inherently hierarchical and the choices we make to govern our lives can be broken into smaller sub-tasks or infinitesimal actions, such as a neuron firing to move an arm an inch. Translating this hierarchy of learning to neural agents has been a long-standing challenge in reinforcement learning, and emerging approaches show increasing promise.