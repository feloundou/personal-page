---
title: Exploring New Depths
date: 2021-02-26T15:41:43.950Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
Over the last two weeks I have been delving into new depths, turning a problem over in my mind for days at a time, without certainty of success. Continuing to probe at an idea in the face of possible failure can be daunting, and I wanted to share some of the tips that helped me overcome the challenges of designing novel solutions.

1. **Make a mental model**

   Assuming that the idea you would like to implement is not too far removed from experience you have collected, you should have a fairly good understanding of what success looks like. If your idea works, what should be the expected results? This is a crucial step in experimentation, and serves to orient you when choosing a direction in which to prove for improvements. This said, remain flexible and aware that your idea of success and success itself may differ by a great margin. You may find it helpful to consult other resources to get a grounded view of ideas previously implemented and the scale/scope of their results. In this process, I usually peruse papers that are closely connected to the canonical principle I am applying, via the [Connected Papers](https://www.connectedpapers.com/) site. 
2. **Iterate quickly**

   In applied research, it is difficult to overemphasize how important rapid iteration is. Not only does it help one attain a stronger understanding of model and experimental dynamics, it kept me ideating rather than ruminating on potential failure. Consistently ask "What if I try ... ?" and follow through. This can be easier said than done, so I would also encourage you to do the following:

   * Write down the idea as you imagine in some form (mathematical formula, flow chart, sketch diagram) to identify all the potential moving parts. Doing so has helped me design a starting experimentation strategy.
   * Find or design an experimental set-up that facilitates new experiments on a whim. For my case, this meant relying on object-oriented programming to create, save and load networks. PyTorch makes it easy to write modular code, and there great libraries on GitHub to help you start! Given that I work primarily in reinforcement learning, my current favorite is: [Spinning Up in RL.](https://spinningup.openai.com/en/latest/)
3. **Log results**

   I am strongly biased towards visual learning, and therefore relying on visualizing new material and ideas to understand them. In several posts, I have mentioned my strong affinity for [Weights and Biases](https://wandb.ai/), but it is simply because all at once it solves several problems as I run through dozens of formulations of experiments all at once: 

   * Groups sets of experiments. I can easily compare and contrast results from my experimental setups.
   * Abstracts away many of the plotting issues one faces when first designing a new experiment. Plus the plots are interactive and do not need to live in large directories.
   * Allows for hyper-parameter sweeps that enable one to plug in several potential values (or a range thereof) and a target variable, and go for tea , while the procedure collects information about your parameters.
   * Save your configurations for each. This can be crucial when reviewing previous experiments.
4. **Ask for help**

   If you are unable to make forward progress, find a resource to assist you. My mentor, the [PyTorch forum](https://discuss.pytorch.org/), and Stack Exchange have been my primary resources, but this is an essential point that could help you take a break, pivot or gather inspiration for a new direction altogether.



Back to my runs. Happy experimenting!