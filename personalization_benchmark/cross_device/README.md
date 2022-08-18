# Cross-device experiments in Motley

This directory contains code to reproduce the cross-device experiments in our
paper
["Motley: Benchmarking Heterogeneity and Personalization in Federated Learning"](https://arxiv.org/abs/2206.09262).

Some pip packages are required by this library, and may need to be installed.
For more details, see `requirements.txt`. We recommend running `pip install
--requirement "requirements.txt"`.

We also require [Bazel](https://www.bazel.build/) in order to run the code.
Please see the guide [here](https://bazel.build/install) for installation
instructions.

The main binaries are located at `finetuning_trainer.py` and
`hypcluster_trainer.py`. Here is an example command:

```
bazel run :finetuning_trainer --
--dataset_name={DATASET NAME}
--experiment_name={EXPERIMENT NAME}
--total_rounds={TOTAL ROUNDS}
--clients_per_train_round={COHORT SIZE}
--train_batch_size={CLIENT BATCH SIZE}
--train_epochs=1
--server_optimizer=adam
--server_learning_rate={SERVER LEARNING RATE}
--server_adam_beta_1=0.9
--server_adam_beta_2=0.99
--server_adam_epsilon={SERVER ADAM EPSILON}
--client_optimizer=sgd
--client_learning_rate={CLIENT LEARNING RATE}
--finetune_optimizer=sgd
--finetune_learning_rate={FINETUNE LEARNING RATE}
--finetune_max_epochs={FINETUNE MAX EPOCHS}
--finetune_last_layer={True/False}
--base_random_seed={RANDOM SEED}
```

Below we describe the hyperparameter grids and the best hyperparameters found
for the three algorithms (FedAvg+Fine-tuning, HypCluster, and local training)
compared in our paper.

## Common hyperparameters

The following hyperparameters are fixed across all experiments. Note that we
focus on *FedAdam* here (i.e., a version of
[generalized FedAvg](https://arxiv.org/abs/2003.00295), where the server
optimizer is [Adam optimizer](https://arxiv.org/abs/1412.6980)) because the
[large-cohort training paper](https://arxiv.org/abs/2106.07820) shows it gives
good performance across different datasets.

-   client_optimizer=['sgd']
-   server_optimizer=['adam']
-   server_adam_beta_1=[0.9]
-   server_adam_beta_2=[0.99]
-   train_epochs=[1] # This is the number of local training epochs performed by
    a client during a round of training.

## Computation resources

Below we summarize the computation resources allocated to run one experiment
(i.e., one point in the the hyperparameter grids) on each dataset. Note that the
actual usage may be smaller than the allocated resources.

-   EMNIST: 80 CPUs
-   StackOverflow: 400 CPUs
-   Landmarks: 16 GPUs
-   TedMulti-EnEs: 2 GPUs

## FedAvg + Fine-tuning hyperparaemeters

Definitions of all the hyperparaemeters for running this algorithm can be found
in `finetuning_trainer.py`. Since FedAvg + Fine-tuning is a two step process,
the hyperparameters contain FedAvg (more specifically, FedAdam as mentioned
above) hyperparameters and fine-tuning hyperparameters.

For EMNIST and StackOverflow, we use the best FedAdam hyperparameters from the
[large-cohort training paper](https://arxiv.org/abs/2106.07820). For Landmarks,
we use the best FedAdam hyperparameters from the
[field guide paper](https://arxiv.org/abs/2107.06917). For TedMulti-EnEs, we
tune the FedAdam hyperparameters from scratch.

We have four fine-tuning hyperparameters. Their names are all started with
`finetune_`, including the optimizer used to fine-tune the model (where we focus
on `sgd`), the fine-tuning learning rate, and whether to only fine-tune the last
layer. We also need to tune the number of local epochs used to fine-tune the
model - this value is automatically found by `finetuning_trainer.py` by
postprocessing the validation metrics. Specifically, we compute the average
validation accuracy of the fine-tuned models after every fine-tuning epoch
(until `finetune_max_epochs`), and then find the best fine-tuning epoch that
gives the highest average validation accuracy. The best fine-tuned epoch will be
in the range [0, `finetune_max_epochs`], so all we need is to set a proper value
for `finetune_max_epochs`.

### EMNIST

Fixed hyperparameters:

-   client_learning_rate=[0.1]
-   server_learning_rate=[0.001]
-   server_adam_epsilon=[0.001]
-   clients_per_train_round=[50]
-   train_batch_size=[20]
-   total_rounds=[1500]
-   valid_clients_per_round=[100]
-   test_clients_per_round=[100]
-   rounds_per_evaluation=[100]
-   rounds_per_checkpoint=[100]
-   finetune_optimzier=[‘sgd’]
-   finetune_max_epochs=[20]

Tuned hyperparameters (best values are highlighted in **bold**):

-   finetune_learning_rate=[ 0.001, 0.003, **0.005**, 0.01, 0.05 ]
-   finetune_last_layer=[ True, **False** ]

### StackOverflow

Fixed hyperparameters:

-   client_learning_rate=[1.0]
-   server_learning_rate=[0.1]
-   server_adam_epsilon=[0.001]
-   clients_per_train_round=[200]
-   train_batch_size=[16]
-   total_rounds=[1500]
-   valid_clients_per_round=[200]
-   test_clients_per_round=[200]
-   rounds_per_evaluation=[100]
-   rounds_per_checkpoint=[100]
-   finetune_optimzier=[‘sgd’]
-   finetune_max_epochs=[20]

Tuned hyperparameters (best values are highlighted in **bold**):

-   finetune_learning_rate=[ **10^(-1.0)**, 10^(-0.6), 10^(-0.2), 10^(0.2),
    10^(0.6), 10^(1.0) ]
-   finetune_last_layer=[ True, **False** ]

### Landmarks

Fixed hyperparameters:

-   client_learning_rate=[0.01]
-   server_learning_rate=[10^(-2.5)]
-   server_adam_epsilon=[10^(-5)]
-   clients_per_train_round=[64]
-   train_batch_size=[16]
-   total_rounds=[30000]
-   valid_clients_per_round=[32]
-   test_clients_per_round=[96]
-   rounds_per_evaluation=[1000]
-   rounds_per_checkpoint=[1000]
-   finetune_optimzier=[‘sgd’]
-   finetune_max_epochs=[10]

Tuned hyperparameters (best values are highlighted in **bold**):

-   finetune_learning_rate=[ 0.0001, 0.001, 0.005, **0.007**, 0.01, 0.03, 0.05 ]
-   finetune_last_layer=[ True, **False** ]

### TedMulti-EnEs

Fixed hyperparameters:

-   clients_per_train_round=[32]
-   train_batch_size=[16]
-   total_rounds=[1500]
-   valid_clients_per_round=[98]
-   test_clients_per_round=[117]
-   rounds_per_evaluation=[30]
-   rounds_per_checkpoint=[50]
-   finetune_optimzier=[‘sgd’]
-   finetune_max_epochs=[20]

Tuned hyperparameters (best values are highlighted in **bold**):

-   client_learning_rate=[ 10^(-2.5), 10^(-2), 10^(-1.5), **10^(-1)**, 10^(-0.5)
    ]
-   server_learning_rate=[ 10^(-2.5), **10^(-2)**, 10^(-1.5), 10^(-1), 10^(-0.5)
    ]
-   server_adam_epsilon=[ **0.001**, 0.00001 ]
-   finetune_learning_rate=[ **0.0005**, 0.0007, 0.001, 0.002, 0.003 ]
-   finetune_last_layer=[ True, **False** ]

## HypCluster hyperparameters

Definitions of all the hyperparameters for running this algorithm can be found
in `hypcluster_trainer.py`. Because HypCluster with random initialization
usually ends up with all clients choosing the same model (i.e., the mode
collapse issue discussed in [personalization benchmarking paper]), we will use
models learned by FedAvg to warmstart HypCluster. Specifically, we will run
FedAvg (with the hyperparameters in `FedAvg + Finetuning` above) for `number of
warmstart fedavg rounds`; repeat this for `num_clusters` times, and use the
models to warmstart HypCluster. See `algorithms/checkpoint_utils.py` for how to
extract the model weights from the saved checkpoint created by FedAvg.

### EMNIST

Fixed hyperparameters:

-   clients_per_train_round=[50]
-   train_batch_size=[20]
-   total_rounds=[1500]
-   valid_clients_per_round=[100]
-   test_clients_per_round=[100]
-   rounds_per_evaluation=[100]
-   rounds_per_checkpoint=[100]
-   number of warmstart fedavg rounds: 100

Tuned hyperparameters (best values are highlighted in **bold**):

-   client_learning_rate=[ 0.01, 0.05, **0.1**, 0.2 ]
-   server_learning_rate=[ 0.0001, 0.0005, **0.001**, 0.002 ]
-   server_adam_epsilon=[ **0.0001**, 0.0005, 0.001, 0.002 ]
-   num_clusters=[ **2**, 3, 4, 5 ]

### StackOverflow

Fixed hyperparameters:

-   clients_per_train_round=[200]
-   train_batch_size=[16]
-   total_rounds=[1500]
-   valid_clients_per_round=[200]
-   test_clients_per_round=[200]
-   rounds_per_evaluation=[100]
-   rounds_per_checkpoint=[100]
-   number of warmstart fedavg rounds: 100

Tuned hyperparameters (best values are highlighted in **bold**):

-   client_learning_rate=[ 0.1, **0.5**, 1.0, 2.0 ]
-   server_learning_rate=[ **0.01**, 0.05, 0.1, 0.2 ]
-   server_adam_epsilon=[ 10^(-5), **10^(-4)**, 10^(-3), 10^(-2) ]
-   num_clusters=[ **2**, 3, 4, 5 ]

### Landmarks

Fixed hyperparameters:

-   clients_per_train_round=[64]
-   train_batch_size=[16]
-   total_rounds=[30000]
-   valid_clients_per_round=[32]
-   test_clients_per_round=[96]
-   rounds_per_evaluation=[1000]
-   rounds_per_checkpoint=[1000]
-   number of warmstart fedavg rounds: 8000

Tuned hyperparameters (best values are highlighted in **bold**):

-   client_learning_rate=[ 10^(-3), **10^(-2.5)**, 10^(-2), 10^(-1.5) ]
-   server_learning_rate=[ 10^(-3.5), **10^(-3)**, 10^(-2.5), 10^(-2) ]
-   server_adam_epsilon=[ 10^(-6), 10^(-5), **10^(-4)**, 10^(-3) ]
-   num_clusters=[ **2**, 3, 4 ]

### TedMulti-EnEs

Fixed hyperparameters:

-   clients_per_train_round=[32]
-   train_batch_size=[16]
-   total_rounds=[1500]
-   valid_clients_per_round=[98]
-   test_clients_per_round=[117]
-   rounds_per_evaluation=[30]
-   rounds_per_checkpoint=[50]
-   num_clusters=[2]
-   number of warmstart fedavg rounds: 100

Tuned hyperparameters (best values are highlighted in **bold**):

-   client_learning_rate=[ 10^(-2.5), 10^(-2), 10^(-1.5), **10^(-1)**, 10^(-0.5)
    ]
-   server_learning_rate=[ 10^(-2.5), **10^(-2)**, 10^(-1.5), 10^(-1), 10^(-0.5)
    ]
-   server_adam_epsilon=[ **0.001**, 0.00001 ]

## Local training hyperparameters

Traning a local model at each client can be done by running
`finetuning_trainer.py` with `total_rounds=0`. Note that what happens is that
every client fine-tunes a random model (sent by the server) locally. As long as
we set a large enough `finetune_max_epochs` (note that the best number of epochs
will be identified based on the validation metrics), this will give the desired
metrics where each client learns a local model without federation.

### EMNIST

-   total_rounds=[0]
-   valid_clients_per_round=[100]
-   test_clients_per_round=[100]
-   finetune_last_layer=[ True, **False** ]
-   finetune_learning_rate=[ 0.001, 0.01, **0.1**, 0.5, 1.0 ]
-   finetune_optimzier=[‘sgd’]
-   finetune_max_epochs=[200]

### StackOverflow

-   total_rounds=[0]
-   valid_clients_per_round=[200]
-   test_clients_per_round=[200]
-   finetune_last_layer=[ True, **False** ]
-   finetune_learning_rate= [ 0.1, **0.5**, 1.0 ]
-   finetune_optimzier=[‘sgd’]
-   finetune_max_epochs=[200]

### Landmarks

-   total_rounds=[0]
-   valid_clients_per_round=[32]
-   test_clients_per_round=[96]
-   finetune_last_layer=[ True, **False** ]
-   finetune_learning_rate=[ 0.0001, 0.001, **0.01**, 0.1 ]
-   finetune_optimzier=[‘sgd’]
-   finetune_max_epochs=[50]

### TedMulti-EnEs

-   total_rounds=[0]
-   valid_clients_per_round=[98]
-   test_clients_per_round=[117]
-   finetune_last_layer=[ **True**, False ]
-   finetune_learning_rate=[ 0.5, **1.0**, 2.0, 3.0 ]
-   finetune_optimzier=[‘sgd’]
-   finetune_max_epochs=[50]
