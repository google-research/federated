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
"""Hyperparameter configuration for centralized training."""

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
flags.DEFINE_string('sql_database', None,
                    'Directory for SQL database, if applicable')

FLAGS = flags.FLAGS


def define_parameters() -> List[Mapping[str, Union[str, int, float]]]:
  """Returns a list of dicts of parameters defining the experiment grid."""
  # Base hyperparameters grid for all experiments

  # Define task-specific hyperparameter flags
  if FLAGS.task == 'cifar100_image':
    task_grid = collections.OrderedDict(  # pylint: disable=unused-variable
        task=['cifar100_image'],
        cifar100_image_num_classes=[10],
        sql_database=[FLAGS.sql_database],
        train_val_ratio_intra_client=[4],  # Required
        unpart_clients_proportion=[0.2],  # Required
        cifar100_image_resnet_layers=[18],
        num_epochs=[200],
        # decay_epochs=[60],
        centralized_shuffle_buffer_size=[50000],
        # lr_decay=[0.2],
        centralized_learning_rate=np.logspace(-2.5, -0.5, num=5),
        # part_clients_subsampling_rate=np.logspace(-1.0, -0.0, num=3),
        batch_size=[50],
    )
    sgd_grid = collections.OrderedDict(
        centralized_optimizer=['sgd'],
        centralized_sgd_momentum=[0.9],
    )
    optimizer_grids = [sgd_grid]
  elif FLAGS.task == 'emnist_character':
    if FLAGS.emnist_character_only_digits:
      task_grid = collections.OrderedDict(
          task=['emnist_character'],
          emnist_character_model=['cnn'],
          emnist_character_only_digits=[True],
          unpart_clients_proportion=[0.2],
          part_clients_subsampling_rate=[0.1],
          centralized_shuffle_buffer_size=[100000],
          centralized_learning_rate=np.logspace(-3.0, -1.0, num=5),
          num_epochs=[200],
          # decay_epochs=[6],
          # lr_decay=[0.2],
          batch_size=[50],
      )
    else:
      task_grid = collections.OrderedDict(
          task=['emnist_character'],
          sql_database=[FLAGS.sql_database],
          emnist_character_model=['cnn'],
          emnist_character_only_digits=[False],
          emnist_character_merge_case=[False],
          unpart_clients_proportion=[0.2],  # Required
          # train_val_ratio_intra_client=[4],  # Required
          num_epochs=[200],
          # decay_epochs=[6],
          # lr_decay=[0.1],
          centralized_shuffle_buffer_size=[100000],
          part_clients_subsampling_rate=[0.1],
          centralized_learning_rate=np.logspace(-3.0, -1.0, num=5),
          batch_size=[50],
      )
    sgd_grid = collections.OrderedDict(
        centralized_optimizer=['sgd'],
        centralized_sgd_momentum=[0.9],
    )
    optimizer_grids = [sgd_grid]

  elif FLAGS.task == 'stackoverflow_word':
    task_grid = collections.OrderedDict(  # pylint: disable=unused-variable
        task=['stackoverflow_word'],
        train_val_ratio_intra_client=[4],  # Required
        part_clients_subsampling_rate=np.logspace(-3.0, -1.0, num=5),
        num_epochs=[100],
        decay_epochs=[30],
        lr_decay=[0.1],
        centralized_shuffle_buffer_size=[100000],
        centralized_learning_rate=np.logspace(-2.5, -0.5, num=5),
        batch_size=[500],
        part_clients_per_eval=[500],
        unpart_clients_per_eval=[500],
        test_clients_for_eval=[500],
        max_elements_per_client=[1000])

    adam_grid = collections.OrderedDict(  # pylint: disable=unused-variable
        centralized_optimizer=['adam'],
        centralized_adam_epsilon=[1e-4],
    )
    optimizer_grids = [adam_grid]
  elif FLAGS.task == 'shakespeare_character':
    task_grid = collections.OrderedDict(
        task=['shakespeare_character'],
        unpart_clients_proportion=[0.2],  # Required
        num_epochs=[30],
        centralized_shuffle_buffer_size=[20000],
        centralized_learning_rate=np.logspace(-3.0, -1.0, num=5),
        batch_size=[20])
    adam_grid = collections.OrderedDict(  # pylint: disable=unused-variable
        centralized_optimizer=['adam'],
        centralized_adam_epsilon=[1e-4],
    )
    optimizer_grids = [adam_grid]
  else:
    raise ValueError('Invalid task name.')

  # Define optimizer-specific grids
  all_grids = []
  for opt_grid in optimizer_grids:
    all_grids.append(config_utils.hyper_grid({
        **task_grid,
        **opt_grid,
    }))

  return list(itertools.chain.from_iterable(all_grids))
