# Large-Cohort Training for Federated Learning

This directory contains source code for performing large-cohort training for
federated learning with different optimizers on various models and tasks. The
code was developed for a paper, "On Large-Cohort Training for Federated
Learning".

Some pip packages are required by this library, and may need to be installed.
For more details, see `requirements.txt`. We recommend running `pip install
--requirement "requirements.txt"`.

We also require [Bazel](https://www.bazel.build/) in order to run the code.
Please see the guide
[here](https://docs.bazel.build/versions/master/install.html) for installation
instructions.

## Directory structure

This directory contains binaries for running federated learning simulations,
with supporting utilities. The `tasks` directory contains libraries for building
datasets and models for supported training tasks.

## Example usage

To run a simulation, you need to use `bazel`. The main binary is located at
`trainer.py`. It can be run using the command:

```
bazel run :trainer --
--task={TASK_NAME}
--iterative_process={FedOpt/FedSGD}
--total_rounds={TOTAL ROUNDS}
--client_batch_size={CLIENT BATCH SIZE}
--server_optimizer={sgd/adam/adagrad/lars/lamb}
--server_learning_rate={SERVER LEARNING RATE}
--clients_per_train_round={COHORT SIZE}
--experiment_name={EXPERIMENT NAME}
--base_random_seed={RANDOM SEED}
```

We have the following tasks that can be set via the `--task` flag:

| Task Name      | Flag          | Model               | Task Summary          |
| -------------- | ------------- | ------------------- | --------------------- |
| CIFAR-100      | cifar100      | ResNet-18 (with     | Image classification  |
:                :               : GroupNorm)          :                       :
| EMNIST         | emnist        | CNN (dropout)       | Alpha-numeric         |
:                :               :                     : character recognition :
| EMNIST-Lite    | emnist-lite   | CNN (no dropout)    | Numeric character     |
:                :               :                     : recognition           :
| Shakespeare    | shakespeare   | RNN (2 LSTM layers) | Next-character        |
:                :               :                     : prediction            :
| Stack Overflow | stackoverflow | RNN (1 LSTM layer)  | Next-word prediction  |

Here, `EMNIST-Lite` is intended to be a more lightweight simulation. It only
uses the first 10 classes of EMNIST (the digits 0-9), and a smaller model. For
more in-depth experimentation, we recommend using `EMNIST`, which uses all 62
alpha-numeric labels of EMNIST and a slightly larger model.

### Basic flag usage

Below we describe some of the flags above in more detail.

*   `--iterative_process`: Must be one of `FedOpt` or `FedSGD`. The latter
    indicates that one wishes to train with `FedSGD`, while the former
    encompasses all other methods.
*   `--total_rounds`: The number of training rounds to perform.
*   `--client_batch_size`: The client batch size.
*   `-server_optimizer`: What optimizer should be applied to the server (see
    discussion below).
*   `clients_per_train_round`: The cohort size used for training. For small
    simulations, a value between 10 and 50 should generally suffice.
*   `base_random_seed`: A random seed used to govern the randomness of the
    simulation. Specifically, it determines the model initialization and which
    clients are sampled at each round.

Extra settings can be configured via additional flags (see `trainer.py` for more
details).

### Using different optimizers

In our work, we use various optimization methods: **FedSGD**, **FedAvg**,
**FedAvgM**, **FedAdagrad**, **FedAdam**, **FedLARS** and **FedLamb**. To
recreate our experimental results for each optimizer, use the following
optimizer-specific flags:

*   **FedSGD**: `--iterative_process=FedSGD --server_optimizer=sgd`
*   **FedAvg**: `--iterative_process=FedOpt --server_optimizer=sgd
    --server_sgd_momentum=0.0`
*   **FedAvgM**: `--iterative_process=FedOpt --server_optimizer=sgd
    --server_sgd_momentum=0.9`
*   **FedAdagrad**: `--iterative_process=FedOpt --server_optimizer=adagrad
    --server_adagrad_initial_accumulator_value=0.0
    --server_adagrad_initial_accumulator_value=0.0
    --server_adagrad_epsilon=0.001`
*   **FedAdam**: `--iterative_process=FedOpt --server_optimizer=adam
    --server_adam_beta_1=0.9 --server_adam_beta_2=0.99
    --server_adam_epsilon=0.001`
*   **FedLARS**: `--iterative_process=FedOpt --server_optimizer=lars
    --server_yogi_initial_accumulator_value=0.0 --server_lars_momentum=0.9
    --server_lars_epsilon=0.001`
*   **FedLamb**: `--iterative_process=FedOpt --server_optimizer=lamb
    --server_lamb_beta_1=0.9 --server_lamb_beta_2=0.99
    --server_lamb_epsilon=0.001`

Note that for all optimizers except `FedSGD`, you will also have to use the
arguments `--client_optimizer=sgd --client_learning_rate={CLIENT LEARNING
RATE}`.

### Other flags

Below we list other flags that can be set to reproduce other aspects of our
experiments.

*   `--rounds_to_double_cohort`: Set to a positive number `x` to use dynamic
    cohort sizes. The cohort size will value every `x` rounds.
*   `--clipping`: Set to `False` to turn adaptive clipping off.
*   `--scaling`: Can be one of `constant`, `sqrt`, and `linear`. Set to
    `constant` by default. Use `sqrt` or `linear` to perform server learning
    rate scaling using that rule.
*   `--user_server_warmup`: If set to `True`, the server learning rate will be
    scaled linearly from 0 to `--server_learning_rate` over the first 100
    training rounds.
