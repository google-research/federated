# Copyright 2021, Google LLC.
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
"""Hyperparameter configuration for federated training."""

import collections
import itertools

from typing import List, Mapping, Union

from absl import flags
import numpy as np

from generalization.experiments import config_utils

flags.DEFINE_enum('task', None, [
    'stackoverflow_word', 'cifar100_image', 'emnist_character',
    'shakespeare_character'
], 'Which task to perform training on.')
flags.DEFINE_boolean('emnist_character_only_digits', False,
                     'Whether to use digits-only version of EMNIST.')
flags.DEFINE_enum(
    'variant', None,
    ['client_batch_size', 'train_clients_per_round', 'client_epochs_per_round'],
    'Which variant to compare. If None, experiment on default settings only.')
flags.DEFINE_string('sql_database', None,
                    'Directory for SQL database, if applicable')

FLAGS = flags.FLAGS


def define_parameters() -> List[Mapping[str, Union[str, int, float]]]:
  """Returns a list of dicts of parameters defining the experiment grid."""
  # Base hyperparameters grid for all experiments.
  base_grid = collections.OrderedDict(
      client_optimizer=['sgd'],
      shared_random_seed=[1],
      rounds_per_checkpoint=[10],
  )
  # Add learning rate scheduling flags
  if FLAGS.lr_schedule_type == 'constant':
    lr_schedule_grid = collections.OrderedDict(client_lr_schedule=['constant'])
  elif FLAGS.lr_schedule_type == 'exp_decay':
    lr_schedule_grid = collections.OrderedDict(
        client_lr_schedule=['exp_decay'],
        client_lr_decay_steps=[600],
        client_lr_decay_rate=[0.2],
        client_lr_staircase=[True],
    )
  elif FLAGS.lr_schedule_type == 'inv_sqrt_decay':
    lr_schedule_grid = collections.OrderedDict(
        client_lr_schedule=['inv_sqrt_decay'],
        client_lr_decay_steps=[1],
        client_lr_decay_rate=[1.0],
        client_lr_staircase=[False],
    )
  else:
    raise ValueError('Unexpected value {!s} for flag --lr_schedule_type'.format(
        FLAGS.lr_schedule_type))
  base_grid.update(lr_schedule_grid)

  if FLAGS.task == 'cifar100_image':
    task_grid = collections.OrderedDict(
        task=['cifar100_image'],
        train_val_ratio_intra_client=[4],  # Required
        cifar100_image_num_classes=[10],
        sql_database=[FLAGS.sql_database],
        cifar100_image_l2_weight_decay=np.logspace(-6.0, -3.0, num=7),
        unpart_clients_proportion=[0.2],  # Required
        cifar100_image_resnet_layers=[18],
        total_rounds=[2001],
        client_learning_rate=np.logspace(-2.0, -0.0, num=5),
        server_learning_rate=np.logspace(-1.5, 0.5, num=5),
        client_batch_size=[20],  # base CBS = 20
        client_epochs_per_round=[1],  # base EPR = 1
        train_clients_per_round=[30],
        rounds_per_eval=[100],
    )
    if FLAGS.variant == 'client_batch_size':
      task_grid['client_batch_size'].extend([1, 2, 5, 10, 50, 100])
    elif FLAGS.variant == 'client_epochs_per_round':
      task_grid['client_epochs_per_round'].extend([2, 5, 10, 20])
    elif FLAGS.variant == 'train_clients_per_round':
      task_grid['train_clients_per_round'].extend([1, 2, 5, 20, 50, 100])
    elif FLAGS.variant is None:
      pass
    else:
      raise ValueError('Invalid variant.')

    sgd_grid = collections.OrderedDict(
        server_optimizer=['sgd'],
        server_sgd_momentum=[0.9],
    )
    optimizer_grids = [sgd_grid]

  elif FLAGS.task == 'emnist_character':
    if FLAGS.emnist_character_only_digits:
      task_grid = collections.OrderedDict(
          task=['emnist_character'],
          emnist_character_model=['cnn'],
          emnist_character_only_digits=[True],
          unpart_clients_proportion=[0.2],  # Required
          total_rounds=[3001],
          client_learning_rate=np.logspace(-2.0, 0.0, num=5),
          server_learning_rate=np.logspace(-2.0, 0.0, num=5),
          client_batch_size=[20],  # base CBS = 20
          client_epochs_per_round=[1],  # base EPR = 1
          train_clients_per_round=[20],  # original CPR = 10
          part_clients_subsampling_rate=np.logspace(-2.0, -0.0, num=5),
          rounds_per_eval=[50],
      )
    else:
      task_grid = collections.OrderedDict(
          task=['emnist_character'],
          emnist_character_model=['resnet18'],
          emnist_character_only_digits=[True],
          emnist_character_merge_case=[False],
          unpart_clients_proportion=[0.2],  # Required
          total_rounds=[3001],
          client_learning_rate=np.logspace(-2.0, 0.0, num=5),
          server_learning_rate=np.logspace(-2.0, 0.0, num=5),
          client_batch_size=[20],  # base CBS = 20
          client_epochs_per_round=[1],  # base EPR = 1
          train_clients_per_round=[10],  # base CPR = 10
          part_clients_subsampling_rate=np.logspace(-2.0, -0.0, num=5),
          rounds_per_eval=[100],
      )

    if FLAGS.variant == 'client_batch_size':
      task_grid['client_batch_size'].extend([1, 2, 5, 10, 50, 100])
    elif FLAGS.variant == 'client_epochs_per_round':
      task_grid['client_epochs_per_round'].extend([2, 5, 10, 20, 50, 100])
    elif FLAGS.variant == 'train_clients_per_round':
      task_grid['train_clients_per_round'].extend([1, 2, 5, 20, 50, 100])
    elif FLAGS.variant is None:
      pass
    else:
      raise ValueError('Invalid variant.')

    sgd_grid = collections.OrderedDict(
        server_optimizer=['sgd'],
        server_sgd_momentum=[0.9],
    )
    optimizer_grids = [sgd_grid]

  elif FLAGS.task == 'stackoverflow_word':
    task_grid = collections.OrderedDict(
        task=['stackoverflow_word'],
        train_val_ratio_intra_client=[4],  # Required
        part_clients_subsampling_rate=[0.01],
        total_rounds=[3001],  # original default = 15000
        client_learning_rate=np.logspace(-3.0, -1.0, num=5),
        server_learning_rate=np.logspace(-2.0, 0.0, num=5),
        client_batch_size=[20],  # original default = 16
        train_clients_per_round=[50],  # original default = 50
        client_epochs_per_round=[1],
        rounds_per_eval=[100],
        part_clients_per_eval=[500],
        unpart_clients_per_eval=[500],
        test_clients_for_eval=[500],
        max_elements_per_client=[1000],
    )

    adam_grid = collections.OrderedDict(
        server_optimizer=['adam'], server_adam_epsilon=[1e-4])
    optimizer_grids = [adam_grid]

    if FLAGS.variant == 'client_batch_size':
      task_grid['client_batch_size'].extend([1, 2, 5, 10, 50, 100])
    elif FLAGS.variant == 'client_epochs_per_round':
      task_grid['client_epochs_per_round'].extend([2, 5, 10, 20])
    elif FLAGS.variant == 'train_clients_per_round':
      task_grid['train_clients_per_round'].extend([5, 10, 20, 100])
    elif FLAGS.variant is None:
      pass
    else:
      raise ValueError('Invalid variant.')

  elif FLAGS.task == 'shakespeare_character':
    task_grid = collections.OrderedDict(
        task=['shakespeare_character'],
        unpart_clients_proportion=[0.2],  # Required
        total_rounds=[1501],  # original default = 15000
        client_learning_rate=np.logspace(-2.0, 0.0, num=3),
        server_learning_rate=np.logspace(-2.0, 0.0, num=3),
        client_batch_size=[5],  # original default = 16
        train_clients_per_round=[10],
        client_epochs_per_round=[1],
        rounds_per_eval=[50],
    )
    adam_grid = collections.OrderedDict(
        server_optimizer=['adam'], server_adam_epsilon=[1e-4])
    optimizer_grids = [adam_grid]

    if FLAGS.variant == 'client_batch_size':
      task_grid['client_batch_size'].extend([1, 2, 10, 20])
    elif FLAGS.variant == 'client_epochs_per_round':
      task_grid['client_epochs_per_round'].extend([2, 5, 10, 20])
    elif FLAGS.variant == 'train_clients_per_round':
      task_grid['train_clients_per_round'].extend([5, 20, 50, 100])
    elif FLAGS.variant is None:
      pass
    else:
      raise ValueError('Invalid variant.')

  else:
    raise ValueError('Invalid task name.')

  all_grids = []
  for opt_grid in optimizer_grids:
    all_grids.append(
        config_utils.hyper_grid({
            **task_grid,
            **opt_grid,
            **base_grid
        }))

  return list(itertools.chain.from_iterable(all_grids))
