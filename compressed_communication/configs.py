# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Task, optimizer, and training configs for compression experiments."""

import collections

import numpy as np

AGGREGATORS = [
    'quantize_entropy_code', 'vote_step_size', 'entropy_cross_entropy', 'drive',
    'one_bit_sgd', 'qsgd', 'terngrad', 'three_lc', 'top_k', 'no_compression',
    'rotation_ablation', 'histogram'
]

ROUNDING_TYPES = ['uniform', 'stochastic', 'dithered']

QUANTIZATION_SCHEDULES = [
    'fixed', 'linear_decay', 'exponential_decay', 'step_decay'
]

NORMALIZATION_TYPES = [
    'constant', 'mean_magnitude', 'max_magnitude', 'dimensionless_norm'
]

ROTATION_TYPES = ['hadamard', 'dft', 'identity']

TASKS = {
    'cifar100_image':
        collections.OrderedDict(
            task=['cifar100_image'],
            train_batch_size=[20],
            eval_batch_size=[100],
        ),
    'emnist_character':
        collections.OrderedDict(
            task=['emnist_character'],
            train_batch_size=[20],
            eval_batch_size=[100],
        ),
    'shakespeare_character':
        collections.OrderedDict(
            task=['shakespeare_character'],
            train_batch_size=[4],
            eval_batch_size=[100],
        ),
    'stackoverflow_tag':
        collections.OrderedDict(
            task=['stackoverflow_tag'],
            train_batch_size=[100],
            eval_batch_size=[100],
            max_train_elements=[1000],
            clients_per_eval_round=[500],
        ),
    'stackoverflow_word':
        collections.OrderedDict(
            task=['stackoverflow_word'],
            train_batch_size=[20],
            eval_batch_size=[100],
            max_train_elements=[1000],
            clients_per_eval_round=[500],
        ),
}

# These learning rates are generally intended only for tuning.
LEARNING_RATE_GRID = np.logspace(-3.0, 1.0, num=5)

OPTIMIZERS = {
    'fedadagrad':
        collections.OrderedDict(
            client_optimizer=['sgd'],
            server_optimizer=['adagrad'],
            server_adagrad_initial_accumulator_value=[0.0],
            server_adagrad_epsilon=[0.001],
        ),
    'fedavg':
        collections.OrderedDict(
            client_optimizer=['sgd'],
            server_optimizer=['sgd'],
            server_sgd_momentum=[0.0],
        ),
    'fedavgm':
        collections.OrderedDict(
            client_optimizer=['sgd'],
            server_optimizer=['sgd'],
            server_sgd_momentum=[0.9],
        ),
    'fedadam':
        collections.OrderedDict(
            client_optimizer=['sgd'],
            server_optimizer=['adam'],
            server_adam_beta_1=[0.9],
            server_adam_beta_2=[0.99],
            server_adam_epsilon=[0.001],
        ),
}

# Nested structure of learning rates, whose leaves are tuples of the form
# (server_learning_rate, client_learning_rate). Tuned for using the
# `quantize_entropy_code` aggregator with "stochastic" `q_type`.
TUNED_LEARNING_RATES = {
    'cifar100_image':
        collections.OrderedDict(
            fedavg=(1.0, 0.1),
            fedadam=(0.01, 0.1),
        ),
    'emnist_character':
        collections.OrderedDict(
            fedavg=(1.0, 0.1),
            fedadam=(0.01, 0.1),
        ),
    'stackoverflow_tag':
        collections.OrderedDict(
            fedavg=(1.0, 10.0),
            fedadam=(1.0, 10.0),
        ),
    'stackoverflow_word':
        collections.OrderedDict(
            fedavg=(1.0, 0.1),
            fedadam=(0.1, 0.1),
        ),
}


def get_optimizer_config(optimizer: str, task: str, use_tuned_lrs: bool):
  """Returns a dictionary configuring an optimizer for a given task."""
  optimizer_config = OPTIMIZERS.get(optimizer)
  if optimizer_config is None:
    raise ValueError('The `optimizer` argument must be one of {}'.format(
        OPTIMIZERS.keys()))

  if use_tuned_lrs:
    server_lr, client_lr = TUNED_LEARNING_RATES[task][optimizer]
    server_learning_rates = [server_lr]
    client_learning_rates = [client_lr]
  else:
    server_learning_rates = [1.0
                            ] if optimizer == 'fedavg' else LEARNING_RATE_GRID
    client_learning_rates = LEARNING_RATE_GRID
  optimizer_config['server_learning_rate'] = server_learning_rates
  optimizer_config['client_learning_rate'] = client_learning_rates
  return optimizer_config
