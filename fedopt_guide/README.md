# FedOpt Guide experiments

## Overview

Implements and experiments with three tasks for "A Field Guide to Federated
Optimization" ([arXiv link](https://arxiv.org/abs/2107.06917)).

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

The following command can be used to train 100 rounds for the Stack Overflow
Transformer task.

```
bazel run stackoverflow_transformer:federated_trainer -- --total_rounds=100
--client_optimizer=sgd --client_learning_rate=0.1 --client_batch_size=16
--server_optimizer=adam --server_learning_rate=0.03 --server_adam_epsilon=0.001
--clients_per_round=10 --client_epochs_per_round=1
--experiment_name=stackoverflow_experiment
```

## Citation

```
@article{fedopt_guide,
  author    = {Jianyu Wang and
               Zachary Charles and
               Zheng Xu and
               Gauri Joshi and
               H. Brendan McMahan and
               Blaise Ag{\"{u}}era y Arcas and
               Maruan Al{-}Shedivat and
               Galen Andrew and
               Salman Avestimehr and
               Katharine Daly and
               Deepesh Data and
               Suhas N. Diggavi and
               Hubert Eichner and
               Advait Gadhikar and
               Zachary Garrett and
               Antonious M. Girgis and
               Filip Hanzely and
               Andrew Hard and
               Chaoyang He and
               Samuel Horvath and
               Zhouyuan Huo and
               Alex Ingerman and
               Martin Jaggi and
               Tara Javidi and
               Peter Kairouz and
               Satyen Kale and
               Sai Praneeth Karimireddy and
               Jakub Kone{\v{c}}n{\'y} and
               Sanmi Koyejo and
               Tian Li and
               Luyang Liu and
               Mehryar Mohri and
               Hang Qi and
               Sashank J. Reddi and
               Peter Richt{\'{a}}rik and
               Karan Singhal and
               Virginia Smith and
               Mahdi Soltanolkotabi and
               Weikang Song and
               Ananda Theertha Suresh and
               Sebastian U. Stich and
               Ameet Talwalkar and
               Hongyi Wang and
               Blake E. Woodworth and
               Shanshan Wu and
               Felix X. Yu and
               Honglin Yuan and
               Manzil Zaheer and
               Mi Zhang and
               Tong Zhang and
               Chunxiang Zheng and
               Chen Zhu and
               Wennan Zhu},
  title     = {A Field Guide to Federated Optimization},
  journal   = {arXiv preprint arXiv:2107.06917},
  year      = {2021}
}
```
