---
title: Multiple Experts, Multiple Objectives
date: 2021-04-12T21:31:58.843Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
At long last, I present my Scholars project, where I engineered a (somewhat primitive) framework that disentangles data containing many behaviors from different experts to learn to steer a model towards one mode of behavior or another.

#### Motivation

The internet is chock full of data that many machine learning models leverage for training. These data, however, are produced by people, entities or organizations that have their own utility functions. These data, can therefore be thought of as being conditional on those utility functions. When our models ingest this large chunk of data wholesale, they tend to assimilate and reproduce behaviors mapping to these utility functions. As researchers and designers, we may want to retain the ability to steer our trained models towards or away from some modes of behavior. Furthermore, as our models grow in capability and are applied to increasingly complex and diverse settings, we may want to steer their behavior to align with the context or human preferences.



#### Proposed Solution

To approach this problem, we start with a reinforcement learning (RL) framework, where policies can be engineered, back-tested and visually inspected. In an offline RL setting, we take a batch of data collected from multiple polices and try to learn from it mode-conditional policies. So in our ideal scenario, this single policy, conditional on the current state and a context vector would correspond exactly to the policy of some expert conditional only on the state.



**Training the Model**

So what is the process for training the model? First, we collect examples, which is straightforward. We chose to work with examples rather than expert policies because examples of success are easier to come by in nature, especially if we think back to our motivation of Internet data. 

Then we pass these examples through a Vector-Quantized Variational Autoencoder (VQ-VAE) that will cluster them. These labels are passed to a generator, which in this context is a Gaussian actor, that will recover a probability distribution based on the current state and proposed cluster label. Let’s take a closer look at the architecture.



**A. The Clustering Network**

Our clustering algorithm here is a VQ-VAE inspired by Oord et al (2017) with a small modification. The VQ-VAE, as usual has the objective of distilling its inputs to latent dimensions well enough that the decoder can reconstruct them. In the middle here is the embedding space, which maps the encoder representations to cluster via a simple argmin function, i.e. if a tensor is closest in euclidean distance to embedding tensor j, then map that sample to cluster j. Note that this means that the dimensionality of the embedding space determines the  maximum total number of clusters you allow the VQ-VAE to create. If you set k=n, then you can get **up to** n clusters. I want to note here that the labels are the most essential component here. Where a traditional VQ-VAE produces discrete labels, instead of the discrete labels, we simply take the distance vectors as they contain a richer signal while the model trains. These distances will become the instructions that we send to the generator to tell it how we want it to behave.



**B. The Policy Network**

Now that we have received cluster labels from the VQ-VAE, we concatenate them with the state and pass them through a Gaussian MLP. From the Gaussian’s Normal Distribution, then, we evaluate the probability of the actions taken by the “expert”, given the state and cluster labels we have assigned. What happens here is that we increase the probabilities of the true action under the conditional normal distribution. What does the training objective look like? I will flash the formula quickly then proceed to explain the highlights.

This is what happens at inference time. The model observes a state, concatenates a context vector supplied by the supervisor (currently me) and then produces a conditional policy and draws an action. Let’s look at some demos.

Here is the training objective. For those who recognize it, the numerator is the VQ-VAE objective, and the denominator is the policy loss.

So what does it do? The first term of the numerator is a reconstruction loss to encourage the encoder and decoder to communicate effectively through good latent representations. Then there is an L2 loss from the encoder output that incentives the encoder to make representations that are close to the embeddings. There is also an L2 loss from the decoder output that incentives the embeddings to stay close to the encoder representations. Lastly, the denominator is the policy loss which makes the actions more likely when the clustering algorithm is more confident about the context. All these moving parts are trained simultaneously.

This is the setting we chose for experimentation. It is a continuous control environment that I chose specifically because we can explicitly design behavior and test the quality of context-specific imitation. In this setting, an agent, the red moving object, lives on this plane where he can navigate to any space using an action vector that selects forward and rotational velocities. There is a goal in green, and as you can see, it resets to a different random location every time it is reached. The purple dots are hazards that are costly to run into and the vase (in aquamarine can be ignored).

Here we have 2 custom designed experts. One who is purely goal-seeking and the other who simply moves forward. You can think of these plots as a bird-eye view of the earlier environment, with the blue dots representing the locations where the goal reset after being reached and the red dots representing the hazards. All the environments in the demonstrations are identical, the only variable thing is the agent’s behavior and subsequently, number of goals reached.

Before we look at the demonstrations, I would like to point out 2 factors that seem to significantly influence expert behavior encoding. One is k, the number of allowed partitions, and the other the step size when calculating state transitions. Here, we are looking at a step size of 1. We can see that with fewer allowed partitions, the model struggles to map different experts to different latent clusters, but starting with k=4, we see significant differentiation.

When we take larger steps to calculate the transitions, however, the model seems to learn to cluster much earlier. You can maybe think of this a case where one agent walks forward all the time and the other agent walks forward for 3 steps then turns. In the one step scenario, the model would have difficulty separating the two agents when they are walking straight for some of the time. A future direction therefore, might be to model long- and short-term dependencies, such as with LSTMs or with attention.

To refresh your memory, here is what these experts look like in this setting. In these demonstrations that follow, everything is kept constant, except the context vector that we provide the mode-conditional agents.

We set k to 4 for this particular experiment, and this what behavior corresponds to the 4 clusters. The model is sort of learning to go forward. It tends to be true across experiments that it will pick up the simplest behavior first, before learning more complex ones.

This is the model after 1000 epochs of training. It learns to sweep in wide circles.

Here, we show that our model learns some of the goal seeking behavior in Mode 2, and in Mode 3 continues to learn to go forward. It seems like Mode 1 and Mode 4 do not map clearly to a distinct behavior, or are averages of the two.

**Next Steps:**

* Longer-term path dependencies (LSTM, attention)
* Generalizable properties
* Quantitative performance guarantees
* Learn when/how to mode-switch from:
* * human feedback
  * other environmental feedback
  * compare performance to other context-aware frameworks
* Experiment with different modalities
* * Language samples
* Interpreting learned modes beyond demos