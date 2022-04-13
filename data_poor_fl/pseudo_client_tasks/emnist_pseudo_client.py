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
"""Creates an EMNIST training task with pseudo-clients."""

import math
from typing import List

import pandas as pd
import tensorflow_federated as tff


from data_poor_fl.pseudo_client_tasks import pseudo_client_data


def _get_pseudo_client_ids(examples_per_pseudo_clients: int,
                           base_client_examples_df: pd.DataFrame,
                           separator: str = '-') -> List[str]:
  """Generates a list of pseudo-client ids."""
  pseudo_client_ids = []
  for _, row in base_client_examples_df.iterrows():
    num_pseudo_clients = math.ceil(row.num_examples /
                                   examples_per_pseudo_clients)
    client_id = row.client_id
    expanded_client_ids = [
        client_id + separator + str(i) for i in range(num_pseudo_clients)
    ]
    pseudo_client_ids += expanded_client_ids
  return pseudo_client_ids


def build_task(
    base_task: tff.simulation.baselines.BaselineTask,
    examples_per_pseudo_client: int) -> tff.simulation.baselines.BaselineTask:
  """Creates an EMNIST task with pseudo-clients.

  This task will use the same model and preprocessing functions as `base_task`,
  but will split the train and test datasets into pseudo-clients.

  Args:
    base_task: A `tff.simulation.baselines.BaselineTask` for some EMNIST task.
    examples_per_pseudo_client: An integer representing the maximum number of
      examples held by each pseudo-client.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  train_csv_file_path = 'data_poor_fl/pseudo_client_tasks/emnist_train_num_examples.csv'
  with open(train_csv_file_path) as train_csv_file:
    train_client_example_counts = pd.read_csv(train_csv_file)
  train_pseudo_client_ids = _get_pseudo_client_ids(examples_per_pseudo_client,
                                                   train_client_example_counts)
  extended_train_data = pseudo_client_data.create_pseudo_client_data(
      base_task.datasets.train_data,
      examples_per_pseudo_client=examples_per_pseudo_client,
      pseudo_client_ids=train_pseudo_client_ids)

  test_csv_file_path = 'data_poor_fl/pseudo_client_tasks/emnist_test_num_examples.csv'
  with open(test_csv_file_path) as test_csv_file:
    test_client_example_counts = pd.read_csv(test_csv_file)
  test_pseudo_client_ids = _get_pseudo_client_ids(examples_per_pseudo_client,
                                                  test_client_example_counts)
  extended_test_data = pseudo_client_data.create_pseudo_client_data(
      base_task.datasets.test_data,
      examples_per_pseudo_client=examples_per_pseudo_client,
      pseudo_client_ids=test_pseudo_client_ids)
  emnist_pseudo_client_datasets = tff.simulation.baselines.BaselineTaskDatasets(
      train_data=extended_train_data,
      validation_data=None,
      test_data=extended_test_data,
      train_preprocess_fn=base_task.datasets.train_preprocess_fn,
      eval_preprocess_fn=base_task.datasets.eval_preprocess_fn)
  emnist_pseudo_client_datasets.get_centralized_test_data = base_task.datasets.get_centralized_test_data
  emnist_pseudo_client_task = tff.simulation.baselines.BaselineTask(
      datasets=emnist_pseudo_client_datasets, model_fn=base_task.model_fn)
  return emnist_pseudo_client_task
