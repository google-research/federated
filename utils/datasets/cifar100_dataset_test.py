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
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from utils.datasets import cifar100_dataset


TEST_DATA = collections.OrderedDict(
    coarse_label=([tf.constant(1, dtype=tf.int64)]),
    image=([tf.zeros((32, 32, 3), dtype=tf.uint8)]),
    label=([tf.constant(1, dtype=tf.int64)]),
)


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
    preprocess_fn = cifar100_dataset.create_preprocess_fn(
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
    preprocess_fn = cifar100_dataset.create_preprocess_fn(
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
    image_map = cifar100_dataset.build_image_map(crop_shape, distort=False)
    cropped_example = image_map(simple_example)

    self.assertEqual(cropped_example[0].shape, crop_shape)
    self.assertAllClose(x, cropped_example[0], rtol=1e-03)
    self.assertEqual(cropped_example[1], 0)


CIFAR100_LOAD_DATA = 'tensorflow_federated.simulation.datasets.cifar100.load_data'


class FederatedDatasetTest(tf.test.TestCase):

  @mock.patch(CIFAR100_LOAD_DATA)
  def test_preprocess_applied(self, mock_load_data):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    # Mock out the actual data loading from disk. Assert that the preprocessing
    # function is applied to the client data, and that only the ClientData
    # objects we desired are used.
    #
    # The correctness of the preprocessing function is tested in other tests.
    mock_train = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_test = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_load_data.return_value = (mock_train, mock_test)

    _, _ = cifar100_dataset.get_federated_datasets()

    mock_load_data.assert_called_once()

    # Assert the training and testing data are preprocessed.
    self.assertEqual(mock_train.mock_calls,
                     mock.call.preprocess(mock.ANY).call_list())
    self.assertEqual(mock_test.mock_calls,
                     mock.call.preprocess(mock.ANY).call_list())

  def test_raises_length_2_crop(self):
    with self.assertRaises(ValueError):
      cifar100_dataset.get_federated_datasets(crop_shape=(32, 32))


class CentralizedDatasetTest(tf.test.TestCase):

  @mock.patch(CIFAR100_LOAD_DATA)
  def test_preprocess_applied(self, mock_load_data):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    # Mock out the actual data loading from disk. Assert that the preprocessing
    # function is applied to the client data, and that only the ClientData
    # objects we desired are used.
    #
    # The correctness of the preprocessing function is tested in other tests.
    sample_ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)

    mock_train = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_train.create_tf_dataset_from_all_clients = mock.Mock(
        return_value=sample_ds)

    mock_test = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_test.create_tf_dataset_from_all_clients = mock.Mock(
        return_value=sample_ds)

    mock_load_data.return_value = (mock_train, mock_test)

    _, _ = cifar100_dataset.get_centralized_datasets()

    mock_load_data.assert_called_once()

    # Assert the validation ClientData isn't used, and the train and test
    # are amalgamated into datasets single datasets over all clients.
    self.assertEqual(mock_train.mock_calls,
                     mock.call.create_tf_dataset_from_all_clients().call_list())
    self.assertEqual(mock_test.mock_calls,
                     mock.call.create_tf_dataset_from_all_clients().call_list())

  def test_raises_length_2_crop(self):
    with self.assertRaises(ValueError):
      cifar100_dataset.get_centralized_datasets(crop_shape=(32, 32))


if __name__ == '__main__':
  tf.test.main()
