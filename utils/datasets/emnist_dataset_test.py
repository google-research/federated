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

import collections

import tensorflow as tf

from utils.datasets import emnist_dataset

NUM_ONLY_DIGITS_CLIENTS = 3383
TOTAL_NUM_CLIENTS = 3400


TEST_DATA = collections.OrderedDict(
    pixels=([tf.zeros((28, 28), dtype=tf.float32)]),
    label=([tf.constant(0, dtype=tf.int32)]),
)


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class DigitRecognitionPreprocessFnTest(tf.test.TestCase):

  def test_preprocess_element_spec(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_dataset.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        mapping_fn=emnist_dataset._reshape_for_digit_recognition)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                      tf.TensorSpec(shape=(None,), dtype=tf.int32)))

  def test_preprocess_returns_correct_element(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_dataset.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        mapping_fn=emnist_dataset._reshape_for_digit_recognition)
    preprocessed_ds = preprocess_fn(ds)

    element = next(iter(preprocessed_ds))
    expected_element = (tf.zeros(shape=(1, 28, 28, 1), dtype=tf.float32),
                        tf.zeros(shape=(1,), dtype=tf.int32))
    self.assertAllClose(self.evaluate(element), expected_element)


class AutoencoderPreprocessFnTest(tf.test.TestCase):

  def test_preprocess_element_spec(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_dataset.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        mapping_fn=emnist_dataset._reshape_for_autoencoder)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, 784), dtype=tf.float32)))

  def test_preprocess_returns_correct_element(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_dataset.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        mapping_fn=emnist_dataset._reshape_for_autoencoder)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, 784), dtype=tf.float32)))

    element = next(iter(preprocessed_ds))
    expected_element = (tf.ones(shape=(1, 784), dtype=tf.float32),
                        tf.ones(shape=(1, 784), dtype=tf.float32))
    self.assertAllClose(self.evaluate(element), expected_element)


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
        train_client_batch_size=2,
        train_client_epochs_per_round=1,
        test_client_batch_size=3,
        test_client_epochs_per_round=1,
        only_digits=True)

    sample_train_ds = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])
    for train_batch in sample_train_ds:
      train_batch_size = train_batch[0].shape[0]
      train_batch_shape = train_batch[0].shape[1:]
      self.assertLessEqual(train_batch_size, 2)
      self.assertEqual(train_batch_shape, [28, 28, 1])

    sample_test_ds = emnist_test.create_tf_dataset_for_client(
        emnist_test.client_ids[0])
    for test_batch in sample_test_ds:
      test_batch_size = test_batch[0].shape[0]
      test_batch_shape = test_batch[0].shape[1:]
      self.assertLessEqual(test_batch_size, 3)
      self.assertEqual(test_batch_shape, [28, 28, 1])

  def test_autoencoder_dataset_shape(self):
    emnist_train, emnist_test = emnist_dataset.get_federated_datasets(
        train_client_batch_size=2,
        train_client_epochs_per_round=1,
        test_client_batch_size=3,
        test_client_epochs_per_round=1,
        only_digits=True,
        emnist_task='autoencoder')

    sample_train_ds = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])
    for train_batch in sample_train_ds:
      self.assertEqual(train_batch[0].shape, train_batch[1].shape)
      train_batch_size = train_batch[0].shape[0]
      train_batch_shape = train_batch[0].shape[1]
      self.assertLessEqual(train_batch_size, 2)
      self.assertEqual(train_batch_shape, 28 * 28)

    sample_test_ds = emnist_test.create_tf_dataset_for_client(
        emnist_test.client_ids[0])
    for test_batch in sample_test_ds:
      self.assertEqual(test_batch[0].shape, test_batch[1].shape)
      test_batch_size = test_batch[0].shape[0]
      test_batch_shape = test_batch[0].shape[1]
      self.assertLessEqual(test_batch_size, 3)
      self.assertEqual(test_batch_shape, 28 * 28)

  def test_raises_negative_client_epochs(self):
    with self.assertRaisesRegex(
        ValueError,
        'train_client_epochs_per_round must be a positive integer.'):
      emnist_dataset.get_federated_datasets(
          train_client_batch_size=10, train_client_epochs_per_round=-1)

    with self.assertRaisesRegex(
        ValueError, 'test_client_epochs_per_round must be a positive integer.'):
      emnist_dataset.get_federated_datasets(
          train_client_batch_size=10,
          train_client_epochs_per_round=1,
          test_client_batch_size=10,
          test_client_epochs_per_round=-1)

  def test_raises_non_emnist_task(self):
    with self.assertRaisesRegex(ValueError, 'emnist_task must be one of'):
      emnist_dataset.get_federated_datasets(emnist_task='non_task')


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

  def test_autoencoder_dataset_shapes(self):
    emnist_train, emnist_test = emnist_dataset.get_centralized_datasets(
        train_batch_size=32,
        test_batch_size=100,
        only_digits=False,
        emnist_task='autoencoder')

    train_batch = next(iter(emnist_train))
    self.assertEqual(train_batch[0].shape, train_batch[1].shape)
    train_batch_shape = train_batch[0].shape

    test_batch = next(iter(emnist_test))
    self.assertEqual(test_batch[0].shape, test_batch[1].shape)
    test_batch_shape = test_batch[0].shape
    self.assertEqual(train_batch_shape, [32, 28 * 28])
    self.assertEqual(test_batch_shape, [100, 28 * 28])

  def test_nonpositive_shuffle_buffer_size(self):
    emnist_train, _ = emnist_dataset.get_centralized_datasets(
        train_batch_size=100,
        test_batch_size=100,
        train_shuffle_buffer_size=-1,
        test_shuffle_buffer_size=-1,
        only_digits=False)

    train_iter1 = iter(emnist_train)
    train_iter2 = iter(emnist_train)
    for _ in range(5):
      batch1 = next(train_iter1)
      batch2 = next(train_iter2)
      self.assertAllClose(batch1, batch2)

  def test_raises_non_emnist_task(self):
    with self.assertRaisesRegex(ValueError, 'emnist_task must be one of'):
      emnist_dataset.get_centralized_datasets(emnist_task='non_task')


if __name__ == '__main__':
  tf.test.main()
