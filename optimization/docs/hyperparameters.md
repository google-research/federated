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

Task                  | Clients per round | Batch size | Total rounds
--------------------- | ----------------- | ---------- | ------------
cifar100_image        | 10                | 20         | 4000
emnist_autoencoder    | 10                | 20         | 3000
emnist_character      | 10                | 20         | 1500
shakespeare_character | 10                | 4          | 1200
stackoverflow_word    | 50                | 16         | 1500
stackoverflor_tag     | 10                | 100        | 1500

### Task-specific hyperparameters

Next, we list task-specific hyperparameters, and the suggested starting values
(which are set by default when running experiments).

**CIFAR-100 Image Classification**

Hyperparameter  | Flag                       | Value
--------------- | -------------------------- | -----
Image crop size | `cifar100_image_crop_size` | 24

**EMNIST Character Recognition**

Hyperparameter | Flag                     | Value
-------------- | ------------------------ | -----
Model type     | `emnist_character_model` | cnn

**Shakespeare Character Prediction**

Hyperparameter            | Flag                                    | Value
------------------------- | --------------------------------------- | -----
Character sequence length | `shakespeare_character_sequence_length` | 80

**Stack Overflow NWP**

Hyperparameter          | Flag                                 | Value
----------------------- | ------------------------------------ | -----
Vocabulary size         | `stackoverflow_word_vocab_size`      | 10000
Sequence length         | `stackoverflow_word_sequence_length` | 20
Validation set size     | `num_validation_examples`            | 10000
Max examples per client | `max_elements_per_user`              | 1000

**Stack Overflow TP**

Hyperparameter          | Flag                                | Value
----------------------- | ----------------------------------- | -----
Vocabulary size         | `stackoverflow_tag_word_vocab_size` | 10000
Number of labels        | `stackoverflow_tag_tag_vocab_size`  | 500
Validation set size     | `num_validation_examples`           | 10000
Max examples per client | `max_elements_per_user`             | 1000

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

Task                  | Best method(s)
--------------------- | ----------------------------
cifar100_image        | FedYogi
emnist_autoencoder    | FedYogi
emnist_character      | FedAdam, FedYogi, FedAvgM
shakespeare_character | FedAdagrad, FedYogi, FedAvgM
stackoverflow_word    | FedAdam, FedYogi
stackoverflow_tag     | FedAdagrad

In general, we found FedYogi to be the most consistently good optimizer.

### Best hyperparameters

For each task and optimizer, we list the best client and server learning rates,
as well as the best epsilon value for the adaptive optimizers. All values are in
base-10 log format (eg. instead of 0.001, we write -3).

**Client learning rates**

Fed...                | Adagrad | Adam | Yogi | AvgM | Avg
--------------------- | ------- | ---- | ---- | ---- | ----
cifar100_image        | -1      | -1.5 | -1.5 | -1.5 | -1
emnist_autoencoder    | 1.5     | 1    | 1    | 0.5  | 1
emnist_character      | -1.5    | -1.5 | -1.5 | -1.5 | -1
shakespeare_character | 0       | 0    | 0    | 0    | 0
stackoverflow_word    | -0.5    | -0.5 | -0.5 | -0.5 | -0.5
stackoverflow_tag     | 2       | 2    | 2    | 2    | 2

We see that in most cases, the client learning rate can be fixed at a
task-level.

**Server learning rates**

Fed...                | Adagrad | Adam | Yogi | AvgM | Avg
--------------------- | ------- | ---- | ---- | ---- | ---
cifar100_image        | -1      | 0    | 0    | 0    | 0.5
emnist_autoencoder    | -1.5    | -1.5 | -1.5 | 0    | 0
shakespeare_character | -1      | -2.5 | -2.5 | -0.5 | 0
shakespeare_character | -0.5    | -2   | -2   | -0.5 | 0
stackoverflow_word    | -1.5    | -2   | -2   | 0    | 0
stackoverflow_tag     | 1       | -0.5 | -0.5 | 0    | 0

**Epsilon values**

Fed...                | Adagrad | Adam | Yogi
--------------------- | ------- | ---- | ----
cifar100_image        | -2      | -1   | -1
emnist_autoencoder    | -3      | -3   | -3
shakespeare_character | -2      | -4   | -4
shakespeare_character | -1      | -3   | -3
stackoverflow_word    | -4      | -5   | -5
stackoverflow_tag     | -2      | -5   | -5
