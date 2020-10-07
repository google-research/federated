# Hyperparameter Selection for Federated Optimization

In this section, we detail the possible hyperparameters configurable in
federated training using `main/federated_trainer.py`, and the best performing
hyperparameters found in the course of running experiments for
[Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295). We also
discuss how we tuned experiments.

## Configuring experiments

Before selecting hyperparameters, one must first configure the federated
training procedure. There are a number of *cross-task* hyperparameters (ie.
parameters that are used across all tasks) that must be set. There are also a
number of *task-specific* parameters.

### Cross-task hyperparameters

We list the possible cross-task hyperparameters below. We also list the values
used when running experiments in
[Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295).

**Cross-task hyperparameters**

Hyperparameter              | Flag                   | Value
--------------------------- | ---------------------- | ---------------
Number of clients per round | `clients_per_round`    | See table below
Number of client epochs     | `client_epochs`        | 1
Client sampling seed        | `client_datasets_seed` | 1
Total rounds                | `total_rounds`         | See table below
Batch size                  | `client_batch_size`    | See table below

**Default values**

Task               | Clients per round | Batch size | Total rounds
------------------ | ----------------- | ---------- | ------------
CIFAR-100          | 10                | 20         | 4000
EMNIST AE          | 10                | 20         | 3000
EMNIST CR          | 10                | 20         | 1500
Shakespeare        | 10                | 4          | 1200
Stack Overflow LR  | 10                | 100        | 1500
Stack Overflow NWP | 50                | 16         | 1500

### Task-specific hyperparameters

Next, we list task-specific hyperparameters, and the suggested starting values
(which are set by default when running experiments).

**CIFAR-100**

Hyperparameter  | Flag                 | Value
--------------- | -------------------- | -----
Image crop size | `cifar100_crop_size` | 24

**EMNIST CR**

Hyperparameter | Flag              | Value
-------------- | ----------------- | -----
Model type     | `emnist_cr_model` | cnn

**Shakespeare**

Hyperparameter            | Flag                          | Value
------------------------- | ----------------------------- | -----
Character sequence length | `shakespeare_sequence_length` | 80

**Stack Overflow LR**

Hyperparameter          | Flag                            | Value
----------------------- | ------------------------------- | -----
Vocabulary size         | `so_lr_vocab_tokens_size`       | 10000
Number of labels        | `so_lr_vocab_tags_size`         | 500
Validation set size     | `so_lr_num_validation_examples` | 10000
Max examples per client | `so_lr_max_elements_per_user`   | 1000

**Stack Overflow NWP**

Hyperparameter                     | Flag                             | Value
---------------------------------- | -------------------------------- | -----
Vocabulary size                    | `so_nwp_vocab_size`              | 10000
Number of out-of-vocabulary tokens | `so_nwp_num_oov_buckets`         | 1
Sequence length                    | `so_nwp_sequence_length`         | 20
Validation set size                | `so_nwp_num_validation_examples` | 10000
Max examples per client            | `so_nwp_max_elements_per_user`   | 1000
Embedding layer size               | `so_nwp_embedding_size`          | 96
LSTM layer size                    | `so_nwp_latent_size`             | 670
Number of LSTM layers              | `so_nwp_num_layers`              | 1

## Configuring optimizers

Next, one should configure the optimizers being used at both the client and
server level. This can be done by setting the flags `client_optimizer` and
`server_optimizer`. By default, we recommend setting the client optimizer by
`client_optimizer=sgd`.

Depending on how the server optimizer is configured, you can derive different
federated optimization methods. A list of some arguments, with the default
values used in
[Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295), are given
below.

Method         | Server Optimizer             | Other flags and default values
-------------- | ---------------------------- | ------------------------------
**FedAvg**     | `--server_optimizer=sgd`     | `--server_sgd_momentum=0.0`
**FedAvgM**    | `--server_optimizer=sgd`     | `--server_sgd_momentum=0.9`
**FedAdagrad** | `--server_optimizer=adagrad` | `--server_adagrad_initial_accumulator_value=0.0`
**FedAdam**    | `--server_optimizer=adam`    | None
**FedYogi**    | `--server_optimizer=yogi`    | `--server_yogi_initial_accumulator_value=0.0`

### Learning rates and epsilon values

For all choices of client and server optimizers, one must also select their
learning rates via the flags `client_learning_rate` and `server_learning_rate`.

For the adaptive methods, **FedAdagrad**, **FedAdam**, and **FedYogi**, one can
also change their *epsilon* value, a numerical stability constant. This can be
set via `server_{adaptive optimizer}_epsilon`. By default we recommend a value
of 0.001.

### Additional optimizer flags

We create flags for each optimizer based on its *placement* (client or server)
and the Keras argument name. For example, if we set `--client_optimizer=sgd`,
then there will be a flag `client_sgd_momentum` corresponding to the momentum
argument in the
[Keras SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD).
In general, we have flags of the form `{placement}_{optimizer}_{arg name}`.

These flags use the default value corresponding to the Keras default. In
particular, **FedAdam** and **FedYogi** have first-order and second-order
hyperparameters `beta1` and `beta2` set by default to 0.9 and 0.999. While these
can be changed for better performance, we did not do so in order to make the
space of hyperparameters manageable.

## Tuning optimizer hyperparameters

To test the various optimizers, we ran experiments with the configurations given
in [`Configuring experiments`](#configuring-experiments). We varied the client
learning rate, server learning rate, and adaptive epsilon values.

To select the best parameters for each task and optimizer, we used the *average
training accuracy over the last 100 rounds*. This is an important point: In
realistic FL scenarios, we do not necesarily have a validation set for the
purposes of tuning. Thus, we tuned using the training accuracy.

Using this tuning method, we determined the best optimizers for each task, and
the best hyperparameters for each optimizer/task pair.

### Best optimizers for each task

Task               | Best method(s)
------------------ | ----------------------------
CIFAR-100          | FedYogi
EMNIST AE          | FedYogi
EMNIST CR          | FedAdam, FedYogi, FedAvgM
Shakespeare        | FedAdagrad, FedYogi, FedAvgM
Stack Overflow LR  | FedAdagrad
Stack Overflow NWP | FedAdam, FedYogi

In general, we found FedYogi to be the most consistently good optimizer.

### Best hyperparameters

For each task and optimizer, we list the best client and server learning rates,
as well as the best epsilon value for the adaptive optimizers. All values are in
base-10 log format (eg. instead of 0.001, we write -3).

**Client learning rates**

Fed...             | Adagrad | Adam | Yogi | AvgM | Avg
------------------ | ------- | ---- | ---- | ---- | ----
CIFAR-100          | -1      | -1.5 | -1.5 | -1.5 | -1
EMNIST AE          | 1.5     | 1    | 1    | 0.5  | 1
EMNIST CR          | -1.5    | -1.5 | -1.5 | -1.5 | -1
Shakespeare        | 0       | 0    | 0    | 0    | 0
Stack Overflow LR  | 2       | 2    | 2    | 2    | 2
Stack Overflow NWP | -0.5    | -0.5 | -0.5 | -0.5 | -0.5

We see that in most cases, the client learning rate can be fixed at a
task-level.

**Server learning rates**

Fed...             | Adagrad | Adam | Yogi | AvgM | Avg
------------------ | ------- | ---- | ---- | ---- | ---
CIFAR-100          | -1      | 0    | 0    | 0    | 0.5
EMNIST AE          | -1.5    | -1.5 | -1.5 | 0    | 0
EMNIST CR          | -1      | -2.5 | -2.5 | -0.5 | 0
Shakespeare        | -0.5    | -2   | -2   | -0.5 | 0
Stack Overflow LR  | 1       | -0.5 | -0.5 | 0    | 0
Stack Overflow NWP | -1.5    | -2   | -2   | 0    | 0

**Epsilon values**

Fed...             | Adagrad | Adam | Yogi
------------------ | ------- | ---- | ----
CIFAR-100          | -2      | -1   | -1
EMNIST AE          | -3      | -3   | -3
EMNIST CR          | -2      | -4   | -4
Shakespeare        | -1      | -3   | -3
Stack Overflow LR  | -2      | -5   | -5
Stack Overflow NWP | -4      | -5   | -5
