# DP-FTRL in Federated Learning

## Overview

Implements and experiments with DP-FTRLM (momentum variant of differential
private follow-the-regularized-leader) in federated learning. See "Practical and
Private (Deep) Learning without Sampling or Shuffling"
([arXiv link](https://arxiv.org/abs/2103.00039)) for algorithmic details. The
code in this folder is used for the StackOverflow experiments in the paper.

This folder is organized as following,

*   Two python drivers `run_emnist` and `run_stackoverflow` for
    [EMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist)
    and
    [StackOverflow](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow).
*   The training loops is defined in `training_loop`. Compared to
    `utils.training_loop`, this customized `training_loop` adds the possibility
    of training by both epochs of clients shuffling and rounds of clients
    sampling.
*   TFF iterative process builder `dp_fedavg` based on TFF
    [`simple_fedavg` example](https://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/examples/simple_fedavg).
    Major changes: accept a customized class as server optimizers; add the
    option of clipping model delta before sending back to server for
    differential privacy.
*   `optimizer_utils` defined several custimized optimizers including a simple
    reimplementation of SGD, differential private SGD with momentum (DP-SGDM)
    and differential private FTRL with momentum (DP-FTRLM).
*   `tree_aggregation`defined the noise accumulation by a tree structure for
    DP-FTRLM.

TODO(b/172867399): add privacy computation method.

## Requirements

This code is implemented with
[TensorFlow Federated](https://www.tensorflow.org/federated). See
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research)
for more instructions on using TensorFlow Federated for research.

The following packages may need to be installed

<!-- mdformat off (multiple lines of small code piece) -->

```bash
absl-py~=0.10
attrs~=19.3.0
numpy~=1.19.2
pandas~=0.24.2
tensorflow-federated-nightly
tf-nightly
```

<!-- mdformat on -->

## Example usage

The following command can be used to reproduce the DP-FTRLM results in Section
5.3 of "Practical and Private (Deep) Learning without Sampling or Shuffling".

```bash
bazel run run_stackoverflow.py --experiment_name=stackoverflow_ftrlm_smalln --server_optimizer=dpftrlm --total_epochs=1 --total_rounds=1600 --client_lr=0.5 --server_lr=3 --clip_norm=1 --noise_multiplier=0.067
```

NOTE: this code version did not implement tree restart for DP-FTRLM. It is not
an exact reproduce when report goal is 1000 (`--clients_per_round=1000`).

## Citation

```
@article{kairouz2021practical,
  title={Practical and Private (Deep) Learning without Sampling or Shuffling},
  author={Kairouz, Peter and McMahan, H Brendan and Song, Shuang and Thakkar, Om and Thakurta, Abhradeep and Xu, Zheng},
  journal={arXiv preprint arXiv:2103.00039},
  year={2021}
}
```
