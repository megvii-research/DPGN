# DPGN: Distribution Propagation Graph Network for Few-shot Learning
This repository contains the public code for the paper: link

<p align='center'>
  <img src='dpgn.png' width="800px">
</p>

## Abstract
Most graph-network-based meta-learning approaches model instance-level relation of examples. We extend this idea further to explicitly model the distribution-level relation of one example to all other examples in a 1-vs-N manner. We propose a novel approach named distribution propagation graph network (DPGN) for few-shot learning. It
conveys both the distribution-level relations and instance-level relations in each few-shot learning task. To combine the distribution-level relations and instance-level relations
for all examples, we construct a dual complete graph network which consists of a point graph and a distribution graph with each node standing for an example. Equipped
with dual graph architecture, DPGN propagates label information from labeled examples to unlabeled examples within several update generations. In extensive experiments on
few-shot learning benchmarks, DPGN outperforms state-of-the-art results by a large margin in 5% ∼ 12% under supervised settings and 7% ∼ 13% under semi-supervised settings. Code will be released.


Code will be released soon.