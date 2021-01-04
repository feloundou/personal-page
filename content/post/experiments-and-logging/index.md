---
title: Experiments and Logging
date: 2020-12-18T17:15:06.152Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
My high-level goal over the last week or so involved finishing the write-up for trajectory collection for constrained and unconstrained agents.

Immediately upon starting training, I butted up against the question of how and when to log experiments and agent performance. Indeed,
it is important to be able to determine which, among the agents one has trained, is considered an "expert." This involves a dual process:

1. saving good hyperparameter sets for future training exercises
2. hand-picking agents from observed performance in the set environment

Out of the box, OpenAI Gym sets the lambda user up with great reporting at the epoch level, but as a visual learner who anticipated lots of 
future experiments, this would be lacking. I decided to adopt Weights and Biases (W&B) into my workflow for two reasons:

1. track experimental configurations and log performance
   		- wandb.config() is the only command you need to log hyperparameter settings and it is compatible with ArgParser
   		- W&B reports immediately give you the ability to mix data with commentary on results and transitions towards newer experimental sets was a top selling point
   for my project
2. the ease with which results can be shared and evaluated

An unexpected boon from this choice was the Hyperparameter sweep functionality, that can be set-up with uncanny ease. Indeed, all you need to do is select the sweep mode: \["grid", "bayes"] and set your parameter values in the acceptable formats.

The output you receive is displayed on the Weights and Biases page and can be downloaded in report form.





| Name          | Runtime |     | Hostname    | Notes | State    | Sweep    | Tags | cost_gamma | cost_lim | gamma | hid |     | l   | seed | steps | steps_per_epoch | cost rate         | cumulative cost | epoch |
| ------------- | ------- | --- | ----------- | ----- | -------- | -------- | ---- | ---------- | -------- | ----- | --- | --- | --- | ---- | ----- | --------------- | ----------------- | --------------- | ----- |
| firstsweep    | 118     |     | tyna-server | \-    | finished | yy302nxi |      | 0.99       | 10       | 0.98  | 64  |     | 2   | 456  | 4000  |                 | 0.099788461538462 | 5189            | 12    |
| firstsweep    | 427     |     | tyna-server | \-    | finished | yy302nxi |      | 0.99       | 10       | 0.98  | 64  |     | 2   | 123  | 4000  |                 | 0.073775          | 14755           | 49    |
| firstsweep    | 411     |     | tyna-server | \-    | finished | yy302nxi |      | 0.99       | 10       | 0.98  | 64  |     | 2   | 99   | 4000  |                 | 0.06565           | 13130           | 49    |
| firstsweep    | 375     |     | tyna-server | \-    | finished | yy302nxi |      | 0.99       | 10       | 0.98  | 64  |     | 1   | 999  | 4000  |                 | 0.07765           | 15530           | 49    |
| firstsweep    | 336     |     | tyna-server | \-    | finished | yy302nxi |      | 0.99       | 10       | 0.98  | 64  |     | 1   | 456  | 4000  |                 | 0.070185          | 14037           | 49    |
| firstsweep    | 361     |     | tyna-server | \-    | finished | yy302nxi |      | 0.99       | 10       | 0.98  | 64  |     | 1   | 99   | 4000  |                 | 0.063705          | 12741           | 49    |
| rose-sweep-1  | 0       |     |             | \-    | crashed  | yy302nxi |      |            |          | 0.98  | 64  |     | 1   | 0    |       |                 |                   |                 |       |
| secondrun     | 360     |     | tyna-server | \-    | finished |          |      |            | 10       | 0.99  | 64  | 64  | 2   | 0    | 4000  | 4000            | 0.075685          | 15137           | 49    |
| pretty-bird-1 | 330     |     | tyna-server | \-    | failed   |          |      |            |          |       |     |     |     | 0    |       |                 | 0.073475          |                 | 49    |