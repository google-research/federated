# Copyright 2019, Google LLC.
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
"""Library for loading and preprocessing EMNIST training and testing data."""

import functools
from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

MAX_CLIENT_DATASET_SIZE = 418


def _reshape_emnist_element(element):
  return (tf.expand_dims(element['pixels'], axis=-1), element['label'])


def _preprocess(dataset,
                shuffle_buffer_size,
                num_epochs,
                batch_size,
                max_batches,
                num_parallel_calls=tf.data.experimental.AUTOTUNE):
  """Preprocessing function for EMNIST client datasets."""
  return dataset.shuffle(shuffle_buffer_size).repeat(num_epochs).batch(
      batch_size, drop_remainder=False).take(max_batches).map(
          _reshape_emnist_element, num_parallel_calls=num_parallel_calls)


def get_federated_datasets(
    train_client_batch_size: Optional[int] = 20,
    test_client_batch_size: Optional[int] = 100,
    train_client_epochs_per_round: Optional[int] = 1,
    test_client_epochs_per_round: Optional[int] = 1,
    max_batches_per_train_client: Optional[int] = -1,
    max_batches_per_test_client: Optional[int] = -1,
    train_shuffle_buffer_size: Optional[int] = MAX_CLIENT_DATASET_SIZE,
    test_shuffle_buffer_size: Optional[int] = 1,
    only_digits: Optional[bool] = False):
  """Loads and preprocesses federated EMNIST training and testing sets.

  Args:
    train_client_batch_size: The batch size for all train clients.
    test_client_batch_size: The batch size for all test clients.
    train_client_epochs_per_round: The number of epochs each train client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. If
      negative, the dataset is repeated indefinitely in which case
      `max_batches_per_train_client` must be a positive integer in order to
      ensure the loop is finite.
    test_client_epochs_per_round: The number of epochs each test client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. If
      negative, the dataset is repeated indefinitely in which case
      `max_batches_per_train_client` must be a positive integer in order to
      ensure the loop is finite.
    max_batches_per_train_client: The maximum number of batches (of size
      `train_client_batch_size`) in each train client's dataset. This is
      enforced via `tf.data.Dataset.take`. If this value is negative, then no
      maximum number of batches is enforced.
    max_batches_per_test_client: The maximum number of batches (of size
      `test_client_batch_size`) in each test client's dataset. This is enforced
      via `tf.data.Dataset.take`. If this value is negative, then no maximum
      number of batches is enforced.
    train_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each train client's dataset. By
      default, this is set to the largest dataset size among all clients. If set
      to some integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each test client's dataset. If set
      to some integer less than or equal to 1, no shuffling occurs.
    only_digits: A boolean representing whether to take the digits-only
      EMNIST-10 (with only 10 labels) or the full EMNIST-62 dataset with digits
      and characters (62 labels). If set to True, we use EMNIST-10, otherwise we
      use EMNIST-62.

  Returns:
    A tuple (emnist_train, emnist_test) of `tff.simulation.ClientData` instances
      representing the federated training and test datasets.
  """

  if train_client_epochs_per_round < 0 and max_batches_per_train_client < 0:
    raise ValueError('The arguments `train_client_epochs_per_round` and '
                     '`max_batches_per_train_client` cannot both be negative.')
  if test_client_epochs_per_round < 0 and max_batches_per_test_client < 0:
    raise ValueError('The arguments `test_client_epochs_per_round` and '
                     '`max_batches_per_test_client` cannot both be negative.')
  if train_shuffle_buffer_size <= 1:
    train_shuffle_buffer_size = 1
  if test_shuffle_buffer_size <= 1:
    test_shuffle_buffer_size = 1

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=only_digits)

  preprocess_train_dataset = functools.partial(
      _preprocess,
      shuffle_buffer_size=train_shuffle_buffer_size,
      num_epochs=train_client_epochs_per_round,
      batch_size=train_client_batch_size,
      max_batches=max_batches_per_train_client)

  preprocess_test_dataset = functools.partial(
      _preprocess,
      shuffle_buffer_size=test_shuffle_buffer_size,
      num_epochs=test_client_epochs_per_round,
      batch_size=test_client_batch_size,
      max_batches=max_batches_per_test_client)

  emnist_train = emnist_train.preprocess(preprocess_train_dataset)
  emnist_test = emnist_test.preprocess(preprocess_test_dataset)
  return emnist_train, emnist_test


def get_centralized_datasets(train_batch_size: Optional[int] = 20,
                             test_batch_size: Optional[int] = 500,
                             max_train_batches: Optional[int] = -1,
                             max_test_batches: Optional[int] = -1,
                             train_shuffle_buffer_size: Optional[int] = 10000,
                             test_shuffle_buffer_size: Optional[int] = 1,
                             only_digits: Optional[bool] = False):
  """Loads and preprocesses centralized EMNIST training and testing sets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    max_train_batches: An integer representing the maximum number of batches in
      the training dataset. If set to a negative integer, all batches are used.
    max_test_batches: An integer representing the maximum number of batches in
      the test dataset. If set to a negative integer, all batches are used.
    train_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the train dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the test dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    only_digits: A boolean representing whether to take the digits-only
      EMNIST-10 (with only 10 labels) or the full EMNIST-62 dataset with digits
      and characters (62 labels). If set to True, we use EMNIST-10, otherwise we
      use EMNIST-62.

  Returns:
    A tuple (train_dataset, test_dataset) of `tf.data.Dataset` instances
    representing the centralized training and test datasets.
  """
  if train_shuffle_buffer_size <= 1:
    train_shuffle_buffer_size = 1
  if test_shuffle_buffer_size <= 1:
    test_shuffle_buffer_size = 1

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=only_digits)

  emnist_train = emnist_train.create_tf_dataset_from_all_clients()
  emnist_test = emnist_test.create_tf_dataset_from_all_clients()

  emnist_train = _preprocess(
      emnist_train,
      shuffle_buffer_size=train_shuffle_buffer_size,
      num_epochs=1,
      batch_size=train_batch_size,
      max_batches=max_train_batches)
  emnist_test = _preprocess(
      emnist_test,
      shuffle_buffer_size=test_shuffle_buffer_size,
      num_epochs=1,
      batch_size=test_batch_size,
      max_batches=max_test_batches)

  return emnist_train, emnist_test
