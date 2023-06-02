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
"""Data preprocessing functions for StackOverflow."""

import collections
import functools
from typing import OrderedDict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from personalization_benchmark.cross_device import constants
from personalization_benchmark.cross_device.datasets import emnist

# More than 93% of the training clients have less than 1000 examples (The
# largest client has 194167 examples). We use 1024 to limit the maximum number
# of examples a client can contribute in each round.
_MAX_ELEMENTS_PER_CLIENT = 1024
# StackOverflow's clients are already dividied into a training set (with 342k
# clients) and a held-out set (with 38k clients). We split the held-out clients
# into 10k/28k for validation/test.
_NUM_VALID_CLIENTS = 10000
_ACCURACY_NAME = 'accuracy_without_out_of_vocab_or_end_of_sentence'


def _creation_date_string_to_integer(date: tf.Tensor) -> tf.Tensor:
  """Converts an ISO date string to an integer.

  The converted integers can be used to sort examples by date. Ignores
  fractional seconds if provided. Assumes standard time offset.

  For example:
    2009-06-15 13:45:30 -> 20090615134530
    2009-06-15 13:45:30.345 -> 20090615134530

  Args:
    date: A `tf.string` tensor representing date in ISO 8601 format. The data
      produced by `tff.simulation.datasets.stackoverflow.load_data` conforms to
      this format.

  Returns:
    A `tf.int64` tensor.
  """
  year = tf.strings.substr(date, 0, 4)
  month = tf.strings.substr(date, 5, 2)
  day = tf.strings.substr(date, 8, 2)
  hour = tf.strings.substr(date, 11, 2)
  minute = tf.strings.substr(date, 14, 2)
  second = tf.strings.substr(date, 17, 2)
  timestamp_string = tf.strings.join([year, month, day, hour, minute, second])
  timestamp_integer = tf.strings.to_number(timestamp_string, out_type=tf.int64)
  return timestamp_integer


def _sort_examples_by_date(
    examples: OrderedDict[str, tf.Tensor]) -> OrderedDict[str, tf.Tensor]:
  """Sorts a batch of dataset elements by increasing creation date."""
  date_integers = _creation_date_string_to_integer(examples['creation_date'])
  sorted_indices = tf.argsort(date_integers, stable=True)
  sorted_examples = collections.OrderedDict()
  for key in examples:
    sorted_examples[key] = tf.gather(examples[key], sorted_indices)
  return sorted_examples


def create_model_and_data(
    num_local_epochs: int, train_batch_size: int, use_synthetic_data: bool
) -> Tuple[
    constants.ModelFnType,
    constants.FederatedDatasetsType,
    constants.ProcessFnType,
    constants.SplitDataFnType,
    str,
]:
  """Creates model, datasets, and processing functions for StackOverflow.

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
      examples are sorted by `date` and split into two equal-sized unbatched
      datasets (a personalization dataset and a test dataset). The
      personalization dataset is used for finetuning the model in
      `finetuning_trainer` or choosing the best model in `hypcluster_trainer`.
    5. The accuracy name key stored in the evaluation metrics. It will be used
      in postprocessing the validation clients and test clients metrics in the
      `finetuning_trainer`.
  """
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=num_local_epochs,
      batch_size=train_batch_size,
      max_elements=_MAX_ELEMENTS_PER_CLIENT)
  task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
      train_client_spec=train_client_spec,
      use_synthetic_data=use_synthetic_data)
  train_data = task.datasets.train_data
  heldout_data = task.datasets.validation_data
  if heldout_data is None:
    raise ValueError('Expected stackoverflow validation data to not be None.')
  if use_synthetic_data:
    valid_client_ids = train_data.client_ids
    test_client_ids = train_data.client_ids
  else:
    random_state = np.random.RandomState(seed=constants.SPLIT_CLIENTS_SEED)
    shuffled_heldout_client_ids = random_state.permutation(
        heldout_data.client_ids).tolist()
    valid_client_ids = shuffled_heldout_client_ids[:_NUM_VALID_CLIENTS]
    test_client_ids = shuffled_heldout_client_ids[_NUM_VALID_CLIENTS:]
  new_client_data_fn = functools.partial(
      tff.simulation.datasets.ClientData.from_clients_and_tf_fn,
      serializable_dataset_fn=heldout_data.serializable_dataset_fn)
  datasets = collections.OrderedDict()
  datasets[constants.TRAIN_CLIENTS_KEY] = train_data
  datasets[constants.VALID_CLIENTS_KEY] = new_client_data_fn(valid_client_ids)
  datasets[constants.TEST_CLIENTS_KEY] = new_client_data_fn(test_client_ids)

  eval_preprocess_fn = task.datasets.eval_preprocess_fn

  @tf.function
  def sort_and_split_data_fn(
      raw_data: tf.data.Dataset) -> OrderedDict[str, tf.data.Dataset]:
    """Sort examples by `date` and split it into two unbatched datasets."""
    sorted_data = raw_data.take(_MAX_ELEMENTS_PER_CLIENT).batch(
        _MAX_ELEMENTS_PER_CLIENT).map(_sort_examples_by_date).unbatch()
    # `tff.learning.build_personalization_eval_computation` expects *unbatched*
    # client-side datasets. Batching is part of user-supplied personalization
    # function.
    processed_data = eval_preprocess_fn(sorted_data).unbatch()
    personalization_data, test_data = emnist.split_half(processed_data)
    final_data = collections.OrderedDict()
    final_data[constants.PERSONALIZATION_DATA_KEY] = personalization_data
    final_data[constants.TEST_DATA_KEY] = test_data
    return final_data

  model_fn = task.model_fn
  train_preprocess_fn = task.datasets.train_preprocess_fn
  if train_preprocess_fn is None:
    train_preprocess_fn = lambda x: x
  return (
      model_fn,
      datasets,
      train_preprocess_fn,
      sort_and_split_data_fn,
      _ACCURACY_NAME,
  )
