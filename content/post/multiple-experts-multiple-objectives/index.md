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
**At long last, I present my Scholars project, where I engineered a framework that disentangles data containing many behaviors from different experts to learn to steer a model towards one mode of behavior or another.**





The internet is chock full of data that many machine learning models leverage for training. These data, however, are produced by people, entities or organizations that have their own utility functions. These data, can therefore be thought of as being conditional on those utility functions. When our models ingest this large chunk of data wholesale, they tend to assimilate and reproduce behaviors mapping to these utility functions. As researchers and designers, we may want to retain the ability to steer our trained models towards or away from some modes of behavior. Furthermore, as our models grow in capability and are applied to increasingly complex and diverse settings, we may want to steer their behavior to align with the context or human preferences.