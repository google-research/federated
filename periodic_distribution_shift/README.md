# Federated Learning from Periodically Shifting Distributions

This directory contains source code for training multi-branch neural networks
using federated k-means augmented by temporal prior (or FedTKM). The code was
developed for a paper, "Diurnal or Nocturnal? Federated Learning from
Periodically Shifting Distributions".

## File organization

`main_trainer.py` is the python binary, which controls the tasks, including the
train/val/test splits, and the training loops.

Subfolder `tasks` defines the tasks. `task_utils.py` defines the flags for all
tasks. We use it to choose and configure the tasks.`dist_shift_task.py` and
`dist_shift_task_data.py` provide utilities for evaluating on different
validation subsets (daytime and nighttime). The definition of the models and the
validation/test sets for the three tasks are given in
`emnist_classification_tasks.py`, `cifar_classification_tasks.py` and
`stackoverflow_nwp_tasks.py` respectively.

The data processing are defined in subfolder `datasets`.
`emnist_preprocessing.py`, `cifar_classification_preprocessing.py` and
`stackoverflow_nwp_preprocessing.py`. `datasets/client_sampling.py` simulates
the periodically shifting distribution during training, by defining the daytime
and nighttime subsets for the tasks, and sampling clients from the two subsets
according to a periodically shifting distribution.
`models/dual_branch_resnet_models.py` defines the dual-branch resnet for CIFAR.
For the other two tasks, the models are defined in the task files.
`keras_utils_dual_branch_kmeans.py` and `keras_utils_dual_branch_kmeans_lm.py`
define the forward pass with k-means.

`fedavg_temporal_kmeans.py` defines the local updates of FedTKM on clients, and
aggregation for server updates. For the local updates, FedTKM first runs
inference steps on training samples to select the branch through majority
voting; then trains the model using the selected branch while optionally adds
label smoothing regularization on the other branch. After that, with another
loop of inference over the local training dataset, it computes the averaged
feature while counting the votes for each cluster, then calculate the feature to
update the cluster center with the most vote. The server will update the model
parameters, k-means cluster centers and the distance scalar for temporial prior.

`train_loop_kmeans.py` defines the taining loop and `federated_evaluation.py`
enables federated evaluation.

## Usage

### Simulating the distribution shift

Argument `period` is the period of the distribution shift. Set `shift_fn` to
either `linear` or `cosine` to specify the function type of the periodical
distribution shift. The argument `shift_p` controls the balance of the two modes
in the periodic shifting distributions. If `shift_p=1`, the data distribution is
balanced on the daytime and nighttime modes. Otherwise, the distribution will be
biased towards one mode.

### Hyperparameters for FedTKM

Set `aggregated_kmeans=True` to use FedTKM. `label_smooth_w` is the weight of
the label smoothing regularization, while `label_smooth_eps` (0 to 1) is the
smoothness. `geo_lr` sets the step size of the geometric update, typically
within the range of `[1e-2, 1e-1]`. Optionally, we can set the function type for
the prior through `prior_fn`, which can be either `linear` or `cosine`. We only
included results with `linear` in the paper. Grid search of hyperparameters are
given in the XManager scripts.

## Example commands

```
# on EMNIST
bazel run main_trainer -- \
--task emnist_character --experiment_name test \
--client_optimizer sgd --client_learning_rate 1e-3 \
--server_optimizer adam --server_learning_rate 0.1 --server_adam_epsilon 1e-4 \
--clients_per_round 10 --client_epochs_per_round 1 \
--client_batch_size 20 \
--total_rounds 2049 \
--rounds_per_checkpoint 1 --client_datasets_random_seed 1 \
--rounds_per_eval 1 --max_elements_per_client 66666 \
--period 32 \
--label_smooth_eps 0.1 --label_smooth_w 0.5 \
--emnist_character_batch_majority_voting \
--feature_dim 128 \
--shift_fn linear --aggregated_kmeans

# on CIFAR
bazel run main_trainer -- \
--task cifar100_10 --experiment_name test \
--client_optimizer sgd --client_learning_rate 1e-3 \
--server_optimizer adam --server_learning_rate 0.1 --server_adam_epsilon 1e-4 \
--clients_per_round 10 --client_epochs_per_round 1 \
--client_batch_size 20 \
--total_rounds 2049 \
--rounds_per_checkpoint 1 --client_datasets_random_seed 1 \
--rounds_per_eval 8 --max_elements_per_client 20 \
--period 32 \
--label_smooth_w 0.1 --label_smooth_eps 0.5 \
--cifar100_10_batch_majority_voting \
--feature_dim 512 \
--shift_fn linear --aggregated_kmeans --rescale_eval --zero_mid \
--alsologtostderr

# on Stack Overflow
bazel run main_trainer -- \
--task stackoverflow_word  --experiment_name test \
--client_optimizer sgd --client_learning_rate 0.01 \
--server_optimizer adam --server_learning_rate 0.01 --server_adam_epsilon 1e-5 \
--clients_per_round 2 --client_epochs_per_round 1 --client_batch_size 2 \
--stackoverflow_word_vocab_size 10000 --stackoverflow_word_sequence_length 20 \
--total_rounds 2049 --rounds_per_checkpoint 1500 \
--client_datasets_random_seed=1 --rounds_per_eval 64 \
--max_elements_per_client 2 \
--period 256 --kmeans_k 2 \
--stackoverflow_word_batch_majority_voting --feature_dim 192 \
--aggregated_kmeans --label_smooth_w 0.25 \
--label_smooth_eps 0. \
--shift_fn linear --interp_power 1. \
--geo_lr 0.02 --stackoverflow_word_use_mixed --rescale_eval \
--clip_norm 1 --prior_fn linear --zero_mid \
--stackoverflow_word_num_val_samples 10 \
```

## Citation

```
@article{zhu2021diurnal,
  title={Diurnal or Nocturnal? Federated Learning from Periodically Shifting Distributions},
  author={Zhu, Chen and Xu, Zheng and Chen, Mingqing and Kone{\v{c}}n{\`y}, Jakub and Hard, Andrew and Goldstein, Tom},
  year={2021}
}
```
