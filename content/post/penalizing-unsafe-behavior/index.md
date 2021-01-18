---
title: Penalizing Unsafe Behavior
date: 2021-01-15T19:42:34.221Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
\maketitle
Proximal Policy Optimization Methods
====================================

For vanilla policy gradients (such REINFORCE), the objective used to
optimize the neural network looks like:

$$L^{PG}(\theta) = \hat{\Aver{E}}_t[log \pi_{\theta}(a_t|s_t) \hat{\Aver{A}}_t]$$

where the $\hat{\Aver{A}}$ could be the discounted return (as in
REINFORCE) or the advantage function (as in GAE) for example. Taking a
gradient ascent step on this loss with respect to the network parameters
incentivizes the agent to pursue actions that led to higher reward.\
The vanilla policy gradient method uses the log probability of your
action $(log \pi(a|s))$ to trace the impact of the actions (i.e. credit
assignment), but you could imagine using another function to do this,
for instance a function that uses the probability of the action under
the current policy $(\pi(a|s))$, divided by the probability of the
action under your previous policy $(\pi_{old}(a|s))$, looking like:

$$r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)}$$

This expression is greater than 1 when when $a_t$ is more probable for
the current policy than it is for the old policy; it will be between 0
and 1 when the $a_t$ is less probable for the current policy than for
the old. So, given a difference in rewards between a current policy and
an old one, the actions that are more probable under the current policy
than the old one receive higher credit for the difference.

If the action is much more probable under the current policy,
$r_t(\theta)$ might be so large it leads to taking extremely
large/catastrophic gradient steps. To stabilize these effects, PPO
proposes 2 mechanisms: PPO-CLIP and PPO-Penalty.

PPO-CLIP
--------

The Clipped Surrogate Objective function is specified as:

$$L^{CLIP}(\theta) = \hat{\Aver{E}}_t[ min(r_t(\theta) \hat{\Aver{A}}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{\Aver{A}}_t]$$

The clipping parameter, $\epsilon$ is chosen to bound the $r_t$ value.
For instance, if $\epsilon$ is 0.2, then $r_t$ can vary between 0.8 and
1.2.

PPO-Penalty
-----------

From the paper, \"Another approach, which can be used as an alternative
to the clipped surrogate objective, or in addition to it, is to use a
penalty on KL divergence, and to adapt the penalty coefficient so that
we achieve some target value of the KL divergence $d_target$ each policy
update.\"

$$L^{KL-Penalty}(\theta) = \hat{\Aver{E}}_t[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)} \hat{\Aver{A}}_t - \beta KL[\pi_{\theta old}(.|s_t), \pi(.|s_t) ]]$$

\FOR{k= 0,1,2...}
\STATE{Collect set of trajectories by running policy $\pi_k = \pi(\theta_k) $ in the environment.}
\STATE{Compute rewards-to-go $\hat{R}_t$ }
\STATE{Compute advantage estimates, $\hat{A}_t$ (using any method) based on the current value function, $V_{\psi_k}$ }
\STATE{Update the policy by maximizing the PPO-penalty objective: 
$$L^{KL-Penalty}(\theta) = \hat{\Aver{E}}_t[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)} \hat{\Aver{A}}_t - \beta KL[\pi_{\theta old}(.|s_t), \pi(.|s_t) ]]  $$ via stochastic gradient descent. }
\STATE{Calculate penalized reward: $$R_{PEN} = R - \zeta Cost $$ }
\STATE{Update Penalty Coefficient $\zeta$, $$\zeta_{k+1} = max(0, \zeta_k + \lambda_{\zeta}(Avg Cost - Cost_{lim} )$$ }
\STATE{Fit value function by regression with MSE loss, also via stochastic gradient descent.}
\ENDFOR
\
