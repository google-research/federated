# Federated Posterior Averaging

This directory contains source code for training models using federated
posterior averaging (or FedPA). The code derives from the `optimization`
subfolder and allows to run all the same types of experiments using different
optimizers, learning rate schedules, models, and tasks. The code was developed
for a paper, "Federated Learning via Posterior Averaging: A New Perspective and
Practical Algorithms" ([arXiv link](https://arxiv.org/abs/2010.05273)).

## Using this directory

You can use this directory in all the same ways as `optimization`, with the only
difference is that generalized federated averaging (FedAvg) is substituted with
federated posterior averaging (FedPA). For installation instructions and
prerequisites please follow
[these instructions](../optimization#using-this-directory).

## Running experiments

Suppose we wish to train a convolutional network on EMNIST for purposes of
character recognition (`emnist_cr`), using FedPA algorithm. Various aspects of
the federated training procedure can be customized via `absl` flags. For
example, from this directory one could run:

```bash
bazel run main:federated_trainer -- \
  --task=emnist_cr \
  --total_rounds=1500 \
  --client_optimizer=sgd \
  --client_learning_rate=0.02 \
  --client_sgd_momentum 0.9 \
  --client_batch_size=20 \
  --server_optimizer=sgd \
  --server_learning_rate=0.5 \
  --server_sgd_momentum=0.9 \
  --clients_per_round=100 \
  --client_epochs_per_round=5 \
  --client_mixin_check_start_round=400 \
  --client_mixin_epochs_per_round=1 \
  --client_shrinkage_rho=0.01 \
  --experiment_name=emnist_fedpa_experiment
```

This will run 1500 communication rounds of FedPA training, using SGD on both the
client (for MCMC sampling) and server, with learning rates of 0.02 and 0.5,
respectively. The experiment uses 100 clients in each round, and performs 5
training epochs on each client's dataset with batch size of 20. The
`experiment_name` flag is for the purposes of writing metrics. FedPA-specific
flags are: - `client_mixin_check_start_round`: specifies the number of rounds
that FedPA runs in the FedAvg regime (i.e., client updates are computed as in
FedAvg), before it starts using local MCMC sampling and computing deltas
differently. - `client_mixin_epochs_per_round`: specifies the number of epochs
at each round used to mixin/burnin the IASG MCMC sampler. -
`client_shrinkage_rho`: the hyperparameter of the shrinkage estimator used for
estimating covariances of the local posterior distributions.

## Tasks and datasets

For more details on tasks and datasets, please refer to
[optimization](../optimization#task-and-dataset-summary).

## Hyperparameters and reproducibility

For more details on hyperparameters used in the original paper, please refer to
[Appendix D](https://arxiv.org/pdf/2010.05273.pdf#page=19) of the paper.
