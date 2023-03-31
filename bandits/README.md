# An Empirical Evaluation of Federated Contextual Bandit Algorithms

## Overview

Code for
["An Empirical Evaluation of Federated Contextual Bandit Algorithms"](https://arxiv.org/abs/2303.10218).

## Requirements

This code is implemented with
[TensorFlow Federated](https://www.tensorflow.org/federated). See
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research)
for more instructions on using TensorFlow Federated for research.

The following packages may need to be installed

<!-- mdformat off (multiple lines of small code piece) -->

```bash
absl-py>=1.0,==1.*
attrs~=21.4
dp-accounting==0.3.0
numpy~=1.21
tensorflow~=2.11.0
tensorflow-privacy==0.8.6
tensorflow-federated~=0.49.0
```

<!-- mdformat on -->

## Example usage

The following command can run an experiment on EMNIST from *scratch* with
differential privacy for one of the settings in
[Figure 4](https://arxiv.org/pdf/2303.10218.pdf).

```
bazel run run_federated -- \
    --task_type=emnist_chars --bandits_type=softmax  \
    --bandits_deploy_freq=200 \
    --bandits_temperature=0.05 \
    --server_optimizer=adam --client_optimizer=sgd \
    --client_learning_rate=0.2 --server_learning_rate=0.002 \
    --clients_per_round=64 \
    --client_batch_size=16 --eval_batch_size=256 \
    --max_concurrent_threads=16 \
    --rounds_per_eval=40 --total_rounds=800 \
    --aggregator_type=dpsgd \
    --clip_norm=0.1 --noise_multiplier=0.1
```

We can choose `task_type` from `{emnist_chars, stackoverflow_tag}` to reproduce
the experiments in the paper. Additional tasks of `{emnist_digits,
emnist10_linear, emnist62_linear}` can be explored, though these codepaths were
not used in the experiments in the paper and are less tested.

We can choose `bandits_type` from `{epsilon_greedy_unweight, falcon, softmax}`
to reproduce the experiments in the paper. Additional research exploration can
use `{epsilon_greedy, epsilon_greedy_ce, epsilon_greedy_ce_unw, supervised_mse,
supervised_ce}`, though these codepaths were not used in the experiments in the
paper and are less tested.

`bandits_epsilon` is the hyperparamter for `epsilon_greedy*` algorithms;
`bandits_mu` and `bandits_gamma` are hyperparameters for the `falcon` algorithm;
`bandits_temperature` is the hyperparameter for the `softmax` algorithm. The
greedy baseline can be achieved by setting `bandits_epsilon=0` with
`epsilon_greedy*`.

Leave `aggregator_type` empty for experiments without differential privacy. And
choose from `{dpsgd, dpftrl}` for [DP-FedAvg](https://arxiv.org/abs/1710.06963)
or [DP-FTRL](https://arxiv.org/abs/2103.00039). Hyperparameters `clip_norm` and
`noise_multiplier` can be tuned when DP is turned on. When DP is disabled, we
would suggest `--adaptive_clipping=True` for stability.

We can use `{init, bandits}` for `dist_shift_type` to enable the *init-shift*
scenario in the [paper](https://arxiv.org/abs/2303.10218). Leave it empty for
the *scratch* and *init* scenarios. `population_client_selection` is used to
separate the pool for training the initial model and the followup training with
bandit algorithms.

For the *init* scenario, we can train an initial model by supervised learning on
a small set of clients

```
bazel run
run_federated -- \ --task_type=stackoverflow_tag --bandits_type=supervised_mse \
--bandits_deploy_freq=1 \ --server_optimizer=adam --client_optimizer=sgd \
--client_learning_rate=0.2 --server_learning_rate=0.05 \ --clients_per_round=64
\ --client_batch_size=16 --eval_batch_size=256 \ --max_concurrent_threads=16
--max_validation_samples=10000 \ --rounds_per_eval=10 --total_rounds=100 \
--population_client_selection=0-100 \ --root_output_dir='/tmp/'
--experiment_name='so_init'
```

For the *init-shift* scenario, we use an additional `--dist_shift_type=init`
flag for pretraining.

Loading the pretrained initial model by `initial_model_path`, we can then run
the bandit algorithms for the *init* scenario.

```
bazel run run_federated -- \ --task_type=stackoverflow_tag
--bandits_type=softmax \ --bandits_deploy_freq=200 \ --bandits_temperature=0.05
\ --server_optimizer=adam --client_optimizer=sgd \ --client_learning_rate=0.05
--server_learning_rate=0.005 \ --clients_per_round=64 \ --client_batch_size=16
--eval_batch_size=256 \ --max_concurrent_threads=16
--max_validation_samples=10000 \ --rounds_per_eval=100 --total_rounds=1500 \
--adaptive_clipping=True \ --population_client_selection=170000-342477 \
--initial_model_path='/tmp/models/so_init'
```

For the *init-shift* scenario, we use an additional `--dist_shift_type=bandits`
flag for distribution shift when running bandit algorithms.

### Hyperparameters for reproducibility

See Table 1-9 in the
[appendix of our paper](https://arxiv.org/pdf/2303.10218.pdf).

## Citation

```
@article{agarwal2023empirical,
  title={An Empirical Evaluation of Federated Contextual Bandit Algorithms},
  author={Agarwal, Alekh and McMahan, H Brendan and Xu, Zheng},
  journal={arXiv preprint arXiv:2303.10218},
  year={2023}
}
```
