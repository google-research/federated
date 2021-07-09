# Federated Optimization

This directory contains source code for evaluating federated learning with
different optimizers on various models and tasks. The code was originally
developed for a paper, "Adaptive Federated Optimization"
([arXiv link](https://arxiv.org/abs/2003.00295)), but has since evolved into a
general library for comparing and benchmarking federated optimization
algorithms.

## Using this directory

This library uses [TensorFlow Federated](https://www.tensorflow.org/federated).
For a more general look at using TensorFlow Federated for research, see
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research).

Some pip packages are required by this library, and may need to be installed:

```
absl-py
attrs
dm-tree
numpy
pandas
tensorflow-federated-nightly
tensorflow-model-optimization
tensorflow-privacy
tensorflow-probability
tensorflow-text-nightly
tf-nightly
tfa-nightly
```

We also require [Bazel](https://www.bazel.build/) in order to run the code.
Please see the guide
[here](https://docs.bazel.build/versions/master/install.html) for installation
instructions.

## Directory structure

This directory is broken up into six task directories. Each task directory
contains task-specific libraries (such as libraries for loading the correct
dataset), as well as libraries for performing federated training. These are in
the `task` folder.

A single binary for running these tasks can be found at `trainer.py`. This
binary will, according to `absl` flags, run any of the six task-specific
federated training libraries.

## Example usage

Suppose we wish to train a convolutional network on EMNIST for purposes of
character recognition (`emnist_character`), using federated optimization.
Various aspects of the federated training procedure can be customized via `absl`
flags. For example, from this directory one could run:

```
bazel run :trainer -- --task=emnist_character --total_rounds=100
--client_optimizer=sgd --client_learning_rate=0.1 --client_batch_size=20
--server_optimizer=sgd --server_learning_rate=1.0 --clients_per_round=10
--client_epochs_per_round=1 --experiment_name=emnist_fedavg_experiment
```

This will run 100 communication rounds of federated training, using SGD on both
the client and server, with learning rates of 0.1 and 1.0 respectively. The
experiment uses 10 clients in each round, and performs 1 training epoch on each
client's dataset. Each client will use a batch size of 10 The `experiment_name`
flag is for the purposes of writing metrics.

To try using Adam at the server, we could instead set `--server_optimizer=adam`.
Other parameters that can be set include the batch size on the clients, the
momentum parameters for various optimizers, and the number of total
communication rounds.

## Task and dataset summary

Below we give a summary of the datasets, tasks, and models used in this
directory.

<!-- mdformat off(This table is sensitive to automatic formatting changes) -->

Task Name | Dataset        | Model                             | Task Summary              |
----------|----------------|-----------------------------------|---------------------------|
cifar100_image | [CIFAR-100](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)      | ResNet-18 (with GroupNorm layers) | Image classification      |
emnist_autoencoder | [EMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)         | Bottleneck network                | Autoencoder               |
emnist_character | [EMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)         | CNN (with dropout)                | Character recognition         |
shakespeare_character | [Shakespeare](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data)    | RNN with 2 LSTM layers            | Next-character prediction |
stackoverflow_word | [Stack Overflow](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) | RNN with 1 LSTM layer             | Next-word prediction      |
stackoverflow_tag | [Stack Overflow](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) | Logistic regression classifier    | Tag prediction            |

<!-- mdformat on -->

## Configuring the federated training process

### Using different optimizers

In our work, we compare 5 primary server optimization methods: **FedAvg**,
**FedAvgM**, **FedAdagrad**, **FedAdam**, and **FedYogi**. The first two use SGD
on the server (with **FedAvgM** using server momentum) and the last three use an
adaptive optimizer on the server. All five use client SGD.

To configure these optimizers, use the following flags:

*   **FedAvg**: `--server_optimizer=sgd`
*   **FedAvgM**: `--server_optimizer=sgd --server_sgd_momentum={momentum value}`
*   **FedAdagrad**: `--server_optimizer=adagrad`
*   **FedAdam**: `--server_optimizer=adam`
*   **FedYogi**: `--server_optimizer=yogi`

For adaptive optimizers, one should also set the numerical stability constant
epsilon (tau in Algorithm 2 of
[Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)). This
parameter can be using the flag `server_{adaptive optimizer}_epsilon`. We
recommend a starting value of 0.001, which worked well across task and
optimizers. For a more in-depth discussion, see
[Hyperparameters and Tuning](docs/hyperparameters.md).

For FedAdagrad and FedYogi, we use implementations of Adagrad and Yogi that
allow one to select the `initial_accumulator_value` (see the Keras documentation
on
[Adagrad](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad)).
For all experiments, we used initial accumulator values of 0 (the value fixed in
the Keras implementation of
[Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)).
While this can be tuned, we recommend focusing on tuning learning rates,
momentum parameters, and epsilon values before tuning this value.

### Hyperparameters and reproducibility

The client learning rate (`client_learning_rate`) and server learning rate
(`server_learning_rate`), can be vital for good performance on a task, as can
optimizer-specific hyperparameters. By default, we create flags for each
optimizer based on its *placement* (client or server) and the Keras argument
name. For example, if we set `--client_optimizer=sgd`, then there will be a flag
`client_sgd_momentum` corresponding to the momentum argument in the
[Keras SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD).
In general, we have flags of the form `{placement}_{optimizer}_{arg name}`.

In addition to the optimizer-specific hyperparameters, there are other
parameters that can be configured via flags, including the batch size
(`batch_size`), the number of participating clients per round
(`clients_per_round`), the number of client epochs (`client_epochs`). We also
have a `client_datasets_random_seed` flag that seeds a pseudo-random function
used to sample clients. All results in
[Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) used seed of
1. Changing this may change convergence behavior, as the sampling order of
clients is important in communication-limited settings.

For more details on hyperparameters and tuning, see
[Hyperparameters and Tuning](docs/hyperparameters.md).
