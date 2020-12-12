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

from utils.datasets import cifar10_dataset


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class DatasetTest(tf.test.TestCase):

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

  def test_no_op_crop_process_cifar_example(self):
    crop_shape = (1, 1, 1, 3)
    x = tf.constant([[[[1.0, -1.0, 0.0]]]])  # Has shape (1, 1, 1, 3), mean 0
    x = x / tf.math.reduce_std(x)  # x now has variance 1
    simple_example = collections.OrderedDict(image=x, label=0)
    image_map = cifar10_dataset.build_image_map(crop_shape, distort=False)
    cropped_example = image_map(simple_example)

    self.assertEqual(cropped_example[0].shape, crop_shape)
    self.assertAllClose(x, cropped_example[0], rtol=1e-03)
    self.assertEqual(cropped_example[1], 0)

  def test_raises_length_2_crop(self):
    with self.assertRaises(ValueError):
      cifar10_dataset.get_federated_datasets(crop_shape=(32, 32))
    with self.assertRaises(ValueError):
      cifar10_dataset.get_centralized_datasets(crop_shape=(32, 32))

  def test_raises_negative_epochs(self):
    with self.assertRaisesRegex(
        ValueError, 'client_epochs_per_round must be a positive integer.'):
      cifar10_dataset.get_federated_datasets(train_client_epochs_per_round=-1)


if __name__ == '__main__':
  tf.test.main()
