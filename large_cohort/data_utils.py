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
"""Data utilities for large cohort experiments."""

import functools
import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow_federated as tff


ClientDataType = tff.simulation.datasets.ClientData


def create_train_validation_split(
    client_data: ClientDataType,
    seed: int = 1) -> Tuple[ClientDataType, ClientDataType]:
  """Partitions client data into training and validation data."""
  num_clients = len(client_data.client_ids)
  # We sort the client ids to guarantee a fixed ordering before shuffling.
  client_ids = sorted(client_data.client_ids)
  np.random.RandomState(seed=seed).shuffle(client_ids)

  # After shuffling, we perform an 80/20 split into train and validation ids.
  num_train_clients = int(np.ceil(0.8 * num_clients))
  train_ids = client_ids[:num_train_clients]
  validation_ids = client_ids[num_train_clients:]

  # We now create `tff.simulation.datasets.ClientData` objects from these ids.
  train_data = client_data.from_clients_and_tf_fn(
      train_ids, client_data.serializable_dataset_fn)
  validation_data = client_data.from_clients_and_tf_fn(
      validation_ids, client_data.serializable_dataset_fn)
  return train_data, validation_data


def create_sampling_fn(
    *,
    seed: int,
    client_ids: List[str],
    clients_per_round: int,
    rounds_to_double_cohort: Optional[int] = None
) -> Callable[[int], List[str]]:
  """Creates deterministic, uniform sampling function of client ids."""
  client_sampling_fn = tff.simulation.build_uniform_sampling_fn(
      sample_range=client_ids, replace=False, random_seed=seed)
  if rounds_to_double_cohort is None:
    return functools.partial(client_sampling_fn, size=clients_per_round)
  elif not (isinstance(rounds_to_double_cohort, int) and
            rounds_to_double_cohort > 0):
    raise ValueError('rounds_to_double_cohort must be `None` or a positive '
                     f'integer. Got {rounds_to_double_cohort}')

  # Wrap `tff.simulation.build_uniform_sampling_fn` such that every
  # `rounds_to_double_cohort`, the `size` argument doubles and we create
  # additional samples and concatenate them.
  def doubling_train_client_sampling_fn(round_num) -> List[str]:
    num_doublings = math.floor(round_num / rounds_to_double_cohort)
    clients_to_sample = clients_per_round * int(math.pow(2, num_doublings))
    if clients_to_sample > len(client_ids):
      # Return the entire population if we've doubled past the entire population
      # size.
      return client_ids
    return client_sampling_fn(round_num, size=clients_to_sample)  # pytype: disable=wrong-keyword-args  # gen-stub-imports

  return doubling_train_client_sampling_fn
