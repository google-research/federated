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

import collections

from absl.testing import parameterized
import tensorflow as tf

from utils.datasets import cifar10_dataset


TEST_DATA = collections.OrderedDict(
    image=([tf.zeros((32, 32, 3), dtype=tf.uint8)]),
    label=([tf.constant(1, dtype=tf.int64)]),
)


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class PreprocessFnTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('crop_shape_1_no_distort', (32, 32, 3), False),
      ('crop_shape_2_no_distort', (28, 28, 3), False),
      ('crop_shape_3_no_distort', (24, 26, 3), False),
      ('crop_shape_1_distort', (32, 32, 3), True),
      ('crop_shape_2_distort', (28, 28, 3), True),
      ('crop_shape_3_distort', (24, 26, 3), True),
  )
  def test_preprocess_element_spec(self, crop_shape, distort_image):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = cifar10_dataset.create_preprocess_fn(
        num_epochs=1,
        batch_size=1,
        shuffle_buffer_size=1,
        crop_shape=crop_shape,
        distort_image=distort_image)
    preprocessed_ds = preprocess_fn(ds)
    expected_element_shape = (None,) + crop_shape
    self.assertEqual(
        preprocessed_ds.element_spec,
        (tf.TensorSpec(shape=expected_element_shape, dtype=tf.float32),
         tf.TensorSpec(shape=(None,), dtype=tf.int64)))

  @parameterized.named_parameters(
      ('crop_shape_1_no_distort', (32, 32, 3), False),
      ('crop_shape_2_no_distort', (28, 28, 3), False),
      ('crop_shape_3_no_distort', (24, 26, 3), False),
      ('crop_shape_1_distort', (32, 32, 3), True),
      ('crop_shape_2_distort', (28, 28, 3), True),
      ('crop_shape_3_distort', (24, 26, 3), True),
  )
  def test_preprocess_returns_correct_element(self, crop_shape, distort_image):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = cifar10_dataset.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        crop_shape=crop_shape,
        distort_image=distort_image)
    preprocessed_ds = preprocess_fn(ds)

    expected_element_shape = (1,) + crop_shape
    element = next(iter(preprocessed_ds))
    expected_element = (tf.zeros(
        shape=expected_element_shape,
        dtype=tf.float32), tf.ones(shape=(1,), dtype=tf.int32))
    self.assertAllClose(self.evaluate(element), expected_element)

  def test_no_op_crop(self):
    crop_shape = (1, 1, 3)
    x = tf.constant([[[1.0, -1.0, 0.0]]])  # Has shape (1, 1, 3), mean 0
    x = x / tf.math.reduce_std(x)  # x now has variance 1
    simple_example = collections.OrderedDict(image=x, label=0)
    image_map = cifar10_dataset.build_image_map(crop_shape, distort=False)
    cropped_example = image_map(simple_example)

    self.assertEqual(cropped_example[0].shape, crop_shape)
    self.assertAllClose(x, cropped_example[0], rtol=1e-03)
    self.assertEqual(cropped_example[1], 0)


class LoadCifarTest(tf.test.TestCase):

  def test_num_clients(self):
    cifar_train, cifar_test = cifar10_dataset.load_cifar10_federated()
    self.assertEqual(len(cifar_train.client_ids), 10)
    self.assertEqual(len(cifar_test.client_ids), 10)

  def test_dataset_length(self):
    cifar_train, cifar_test = cifar10_dataset.load_cifar10_federated()
    self.assertEqual(
        _compute_length_of_dataset(
            cifar_train.create_tf_dataset_for_client('0')), 5000)
    self.assertEqual(
        _compute_length_of_dataset(
            cifar_test.create_tf_dataset_for_client('0')), 1000)

  def test_dataset_length_100_clients(self):
    cifar_train, cifar_test = cifar10_dataset.load_cifar10_federated(
        num_clients=100)
    self.assertEqual(
        _compute_length_of_dataset(
            cifar_train.create_tf_dataset_for_client('0')), 500)
    self.assertEqual(
        _compute_length_of_dataset(
            cifar_test.create_tf_dataset_for_client('0')), 100)

  def test_dataset_length_8_clients(self):
    cifar_train, cifar_test = cifar10_dataset.load_cifar10_federated(
        num_clients=8)
    self.assertEqual(
        _compute_length_of_dataset(
            cifar_train.create_tf_dataset_for_client('0')), 6250)
    self.assertEqual(
        _compute_length_of_dataset(
            cifar_test.create_tf_dataset_for_client('0')), 1250)


class FederatedDatasetTest(tf.test.TestCase):

  def test_get_federated_datasets(self):
    cifar_train, cifar_test = cifar10_dataset.get_federated_datasets(
        train_client_batch_size=20, test_client_batch_size=100)
    self.assertEqual(len(cifar_train.client_ids), 10)
    self.assertEqual(len(cifar_test.client_ids), 10)
    self.assertEqual(
        _compute_length_of_dataset(
            cifar_train.create_tf_dataset_for_client('0')), 250)
    self.assertEqual(
        _compute_length_of_dataset(
            cifar_test.create_tf_dataset_for_client('0')), 10)

  def test_federated_cifar_structure(self):
    crop_shape = (28, 28, 3)
    cifar_train, cifar_test = cifar10_dataset.get_federated_datasets(
        train_client_batch_size=3,
        test_client_batch_size=5,
        crop_shape=crop_shape)

    sample_train_ds = cifar_train.create_tf_dataset_for_client(
        cifar_train.client_ids[0])
    train_batch = next(iter(sample_train_ds))
    train_batch_shape = tuple(train_batch[0].shape)
    self.assertEqual(train_batch_shape, (3, 28, 28, 3))

    sample_test_ds = cifar_test.create_tf_dataset_for_client(
        cifar_test.client_ids[0])
    test_batch = next(iter(sample_test_ds))
    test_batch_shape = tuple(test_batch[0].shape)
    self.assertEqual(test_batch_shape, (5, 28, 28, 3))

  def test_raises_length_2_crop(self):
    with self.assertRaises(ValueError):
      cifar10_dataset.get_federated_datasets(crop_shape=(32, 32))

  def test_raises_negative_epochs(self):
    with self.assertRaisesRegex(
        ValueError, 'client_epochs_per_round must be a positive integer.'):
      cifar10_dataset.get_federated_datasets(train_client_epochs_per_round=-1)


class CentralizedDatasetTest(tf.test.TestCase):

  def test_raises_length_2_crop(self):
    with self.assertRaises(ValueError):
      cifar10_dataset.get_centralized_datasets(crop_shape=(32, 32))

  def test_centralized_cifar_structure(self):
    crop_shape = (24, 24, 3)
    cifar_train, cifar_test = cifar10_dataset.get_centralized_datasets(
        train_batch_size=20, test_batch_size=100, crop_shape=crop_shape)
    train_batch = next(iter(cifar_train))
    train_batch_shape = tuple(train_batch[0].shape)
    self.assertEqual(train_batch_shape, (20, 24, 24, 3))
    test_batch = next(iter(cifar_test))
    test_batch_shape = tuple(test_batch[0].shape)
    self.assertEqual(test_batch_shape, (100, 24, 24, 3))


if __name__ == '__main__':
  tf.test.main()
