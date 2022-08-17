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
"""Data preprocessing functions for EMNIST."""

import collections
import functools
from typing import OrderedDict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from personalization_benchmark.cross_device import constants

_SHUFFLE_BUFFER_SIZE = 418  # The largest client has at most 418 examples.
# EMNIST has 3400 clients. We split into 2500/400/500 for train/validation/test.
_NUM_VALID_CLIENTS = 400
_NUM_TEST_CLIENTS = 500
_ACCURACY_NAME = 'sparse_categorical_accuracy'


def split_half(ds: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Splits a dataset into two equal-sized datasets."""
  num_elements_total = ds.reduce(0, lambda x, _: x + 1)
  num_elements_half = tf.cast(num_elements_total / 2, dtype=tf.int64)
  first_data = ds.take(num_elements_half)
  second_data = ds.skip(num_elements_half)
  return first_data, second_data


def create_model_and_data(
    num_local_epochs: int, train_batch_size: int, use_synthetic_data: bool
) -> Tuple[constants.ModelFnType, constants.FederatedDatasetsType,
           constants.ProcessFnType, constants.SplitDataFnType, str]:
  """Creates model, datasets, and processing functions for EMNIST.

  Args:
    num_local_epochs: Number of local epochs performed by a client in a round of
      FedAvg training. It is used in the returned `train_preprocess_fn`, which
      contains an operation `train_dataset.repeat(num_local_epochs)`.
    train_batch_size: Batch size used in FedAvg training. It is also used in the
      returned `train_preprocess_fn`, which contains an operation
      `train_dataset.batch(train_batch_size)`.
    use_synthetic_data: Whether to use synthetic data. This should only be set
      to True for debugging and testing purposes.

  Returns:
    1. A no-argument function that returns a `tff.learning.Model`.
    2. An `OrderedDict` of three federated datasets with keys being
      `constants.TRAIN_CLIENTS_KEY`, `constants.VALID_CLIENTS_KEY` and
      `constants.TEST_CLIENT_KEY`.
    3. A dataset preprocess function for per-client dataset in FedAvg training.
      Batching and repeat operations are contained.
    4. A dataset split function for per-client dataset in evaluation (including
      both validation clients and test clients). Specifically, each client's
      dataset is shuffled and split into two equal-sized unbatched datasets (a
      personalization dataset and a test dataset). The personalization dataset
      is used for finetuning the model in `finetuning_trainer` or choosing the
      best model in `hypcluster_trainer`.
    5. The accuracy name key stored in the evaluation metrics. It will be used
      in postprocessing the validation clients and test clients metrics in the
      `finetuning_trainer`.
  """
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=num_local_epochs,
      batch_size=train_batch_size,
      shuffle_buffer_size=_SHUFFLE_BUFFER_SIZE)
  task = tff.simulation.baselines.emnist.create_character_recognition_task(
      train_client_spec=train_client_spec,
      model_id='cnn',
      only_digits=False,
      use_synthetic_data=use_synthetic_data)
  raw_client_data = task.datasets.train_data
  if use_synthetic_data:
    train_client_ids = raw_client_data.client_ids
    valid_client_ids = raw_client_data.client_ids
    test_client_ids = raw_client_data.client_ids
  else:
    random_state = np.random.RandomState(seed=constants.SPLIT_CLIENTS_SEED)
    shuffled_client_ids = random_state.permutation(
        raw_client_data.client_ids).tolist()
    valid_client_ids = shuffled_client_ids[:_NUM_VALID_CLIENTS]
    test_client_ids = shuffled_client_ids[_NUM_VALID_CLIENTS:(
        _NUM_VALID_CLIENTS + _NUM_TEST_CLIENTS)]
    train_client_ids = shuffled_client_ids[(_NUM_VALID_CLIENTS +
                                            _NUM_TEST_CLIENTS):]
  new_client_data_fn = functools.partial(
      tff.simulation.datasets.ClientData.from_clients_and_tf_fn,
      serializable_dataset_fn=raw_client_data.serializable_dataset_fn)
  datasets = collections.OrderedDict()
  datasets[constants.TRAIN_CLIENTS_KEY] = new_client_data_fn(train_client_ids)
  datasets[constants.VALID_CLIENTS_KEY] = new_client_data_fn(valid_client_ids)
  datasets[constants.TEST_CLIENTS_KEY] = new_client_data_fn(test_client_ids)

  eval_preprocess_fn = task.datasets.eval_preprocess_fn

  @tf.function
  def split_data_fn(
      raw_data: tf.data.Dataset) -> OrderedDict[str, tf.data.Dataset]:
    """Process the raw data and split it equally into two unbatched datasets."""
    # `tff.learning.build_personalization_eval` expects *unbatched* client-side
    # datasets. Batching is part of user-supplied personalization function.
    processed_data = eval_preprocess_fn(raw_data).unbatch().shuffle(
        _SHUFFLE_BUFFER_SIZE,
        seed=constants.SPLIT_CLIENTS_SEED,
        reshuffle_each_iteration=False)
    personalization_data, test_data = split_half(processed_data)
    final_data = collections.OrderedDict()
    final_data[constants.PERSONALIZATION_DATA_KEY] = personalization_data
    final_data[constants.TEST_DATA_KEY] = test_data
    return final_data

  model_fn = task.model_fn
  train_preprocess_fn = task.datasets.train_preprocess_fn
  return model_fn, datasets, train_preprocess_fn, split_data_fn, _ACCURACY_NAME
