# LiDP Auditing: Unleashing the Power of Randomization in Auditing DP

This is the code to reproduce the experimental results of the NeurIPS 2023 paper
[Unleashing the Power of Randomization in Auditing Differentially Private ML](https://arxiv.org/abs/2305.18447).

Auditing differential privacy for ML involves running membership inference many
times and giving high-confidence estimates on the success of the attack (i.e.,
we try to detect the presence of a crafted datapoint, called a "canary" in the
training data).

[This paper](\(https://arxiv.org/abs/2305.18447\)) introduces a variant of DP
called "Lifted DP" (or "LiDP" in short) that is equivalent to the usual notions
of DP. It also gives a recipe to audit LiDP with multiple randomized hypothesis
tests and adaptive confidence intervals to improve the sample complexity of
auditing DP by 4 to 16 times.

## Cite

If you found this code useful, please cite the following work.

```
@incollection{pillutla-etal:lidp_auditing:neurips2023,
title = {{Unleashing the Power of Randomization in Auditing
          Differentially Private ML}},
author = {Krishna Pillutla and Galen Andrew and Peter Kairouz and
          H. Brendan McMahan and Alina Oprea and Sewoong Oh},
booktitle = {NeurIPS},
year = {2023},
}
```
