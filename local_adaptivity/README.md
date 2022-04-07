# Local Adaptivity in FL

## Overview

Implements and experiments with client adaptive optimizers in federated
learning. See "Local Adaptivity in Federated Learning: Convergence and
Consistency" ([arXiv link](https://arxiv.org/abs/2106.02305)) for algorithmic
details.

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
tensorflow-federated~=0.20.0
tensorflow~=2.8.0
```

<!-- mdformat on -->

## Example usage

The following command can be used to train 100 rounds for Stack Overflow Next
Word Prediction task with client AdaGrad and the proposed joint correction
method.

```
bazel run :federated_trainer -- --task=stackoverflow_word --total_rounds=100
--client_optimizer=adagrad --client_learning_rate=0.3 --client_batch_size=20
--server_optimizer=adam --server_learning_rate=0.03 --clients_per_round=10
--client_epochs_per_round=1 --experiment_name=stackoverflow_word_experiment
--client_adagrad_epsilon=0.001 --correction_type=joint
```

## Citation

```
@article{wang2021local,
  title={Local Adaptivity in Federated Learning: Convergence and Consistency},
  author={Wang, Jianyu and Xu, Zheng and Garrett, Zachary and Charles, Zachary and Liu, Luyang and Joshi, Gauri},
  journal={arXiv preprint arXiv:2106.02305},
  year={2021}
}
```
