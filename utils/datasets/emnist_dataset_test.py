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

import tensorflow as tf

from utils.datasets import emnist_dataset

NUM_ONLY_DIGITS_CLIENTS = 3383
TOTAL_NUM_CLIENTS = 3400


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class FederatedDatasetTest(tf.test.TestCase):

  def test_num_clients(self):
    emnist_train, emnist_test = emnist_dataset.get_federated_datasets(
        train_client_batch_size=10,
        train_client_epochs_per_round=1,
        only_digits=True)
    self.assertEqual(len(emnist_train.client_ids), NUM_ONLY_DIGITS_CLIENTS)
    self.assertEqual(len(emnist_test.client_ids), NUM_ONLY_DIGITS_CLIENTS)

    emnist_train, emnist_test = emnist_dataset.get_federated_datasets(
        train_client_batch_size=10,
        train_client_epochs_per_round=1,
        only_digits=False)
    self.assertEqual(len(emnist_train.client_ids), TOTAL_NUM_CLIENTS)
    self.assertEqual(len(emnist_test.client_ids), TOTAL_NUM_CLIENTS)

  def test_dataset_shape(self):
    emnist_train, emnist_test = emnist_dataset.get_federated_datasets(
        train_client_batch_size=5,
        train_client_epochs_per_round=-1,
        max_batches_per_train_client=10,
        test_client_batch_size=6,
        test_client_epochs_per_round=-1,
        max_batches_per_test_client=10,
        only_digits=True)

    sample_train_ds = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])
    for train_batch in sample_train_ds:
      train_batch_shape = train_batch[0].shape
      self.assertEqual(train_batch_shape, [5, 28, 28, 1])

    sample_test_ds = emnist_test.create_tf_dataset_for_client(
        emnist_test.client_ids[0])
    for test_batch in sample_test_ds:
      test_batch_shape = test_batch[0].shape
      self.assertEqual(test_batch_shape, [6, 28, 28, 1])

  def test_take_without_repeat(self):
    emnist_train, emnist_test = emnist_dataset.get_federated_datasets(
        train_client_batch_size=10,
        train_client_epochs_per_round=1,
        max_batches_per_train_client=3,
        test_client_batch_size=10,
        test_client_epochs_per_round=1,
        max_batches_per_test_client=2,
        only_digits=True)

    for i in range(10):
      client_ds = emnist_train.create_tf_dataset_for_client(
          emnist_train.client_ids[i])
      self.assertLessEqual(_compute_length_of_dataset(client_ds), 3)

    for i in range(10):
      client_ds = emnist_test.create_tf_dataset_for_client(
          emnist_test.client_ids[i])
      self.assertLessEqual(_compute_length_of_dataset(client_ds), 2)

  def test_take_with_repeat(self):
    emnist_train, emnist_test = emnist_dataset.get_federated_datasets(
        train_client_batch_size=10,
        train_client_epochs_per_round=-1,
        max_batches_per_train_client=3,
        test_client_batch_size=10,
        test_client_epochs_per_round=-1,
        max_batches_per_test_client=2,
        only_digits=True)

    for i in range(10):
      client_ds = emnist_train.create_tf_dataset_for_client(
          emnist_train.client_ids[i])
      self.assertEqual(_compute_length_of_dataset(client_ds), 3)

    for i in range(10):
      client_ds = emnist_test.create_tf_dataset_for_client(
          emnist_test.client_ids[i])
      self.assertEqual(_compute_length_of_dataset(client_ds), 2)

  def test_raises_no_repeat_and_no_take(self):
    with self.assertRaisesRegex(
        ValueError, 'The arguments `train_client_epochs_per_round` and '
        '`max_batches_per_train_client` cannot both be negative.'):
      emnist_dataset.get_federated_datasets(
          train_client_batch_size=10,
          train_client_epochs_per_round=-1,
          max_batches_per_train_client=-1)

    with self.assertRaisesRegex(
        ValueError, 'The arguments `test_client_epochs_per_round` and '
        '`max_batches_per_test_client` cannot both be negative.'):
      emnist_dataset.get_federated_datasets(
          train_client_batch_size=10,
          train_client_epochs_per_round=1,
          test_client_batch_size=10,
          test_client_epochs_per_round=-1,
          max_batches_per_test_client=-1)


class CentralizedDatasetTest(tf.test.TestCase):

  def test_dataset_shapes(self):
    emnist_train, emnist_test = emnist_dataset.get_centralized_datasets(
        train_batch_size=32, test_batch_size=100, only_digits=False)

    train_batch = next(iter(emnist_train))
    train_batch_shape = train_batch[0].shape
    test_batch = next(iter(emnist_test))
    test_batch_shape = test_batch[0].shape
    self.assertEqual(train_batch_shape, [32, 28, 28, 1])
    self.assertEqual(test_batch_shape, [100, 28, 28, 1])

  def test_take_max_batches(self):
    emnist_train, emnist_test = emnist_dataset.get_centralized_datasets(
        train_batch_size=100,
        test_batch_size=100,
        max_train_batches=3,
        max_test_batches=2,
        only_digits=False)

    self.assertEqual(_compute_length_of_dataset(emnist_train), 3)
    self.assertEqual(_compute_length_of_dataset(emnist_test), 2)

  def test_nonpositive_shuffle_buffer_size(self):
    emnist_train, _ = emnist_dataset.get_centralized_datasets(
        train_batch_size=100,
        test_batch_size=100,
        max_train_batches=5,
        max_test_batches=5,
        train_shuffle_buffer_size=-1,
        test_shuffle_buffer_size=-1,
        only_digits=False)

    train_iter1 = iter(emnist_train)
    train_iter2 = iter(emnist_train)
    for _ in range(5):
      batch1 = next(train_iter1)
      batch2 = next(train_iter2)
      self.assertAllClose(batch1, batch2)


if __name__ == '__main__':
  tf.test.main()
