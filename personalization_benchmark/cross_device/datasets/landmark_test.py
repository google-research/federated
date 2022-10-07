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
import collections
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from personalization_benchmark.cross_device import constants
from personalization_benchmark.cross_device.datasets import landmark


class LandmarkTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_synthetic(self):
    synthetic_cliendata = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        client_ids=['synthetic'],
        serializable_dataset_fn=lambda _: landmark._get_synthetic_landmark_data(  # pylint:disable=g-long-lambda
        ))
    self.assertLen(synthetic_cliendata.client_ids, 1)
    synthetic_image_shape = (600, 800, 3)
    expected_element_type = collections.OrderedDict([
        ('image/decoded',
         tf.TensorSpec(shape=synthetic_image_shape, dtype=tf.uint8)),
        ('class', tf.TensorSpec(shape=(1,), dtype=tf.int64))
    ])
    self.assertEqual(synthetic_cliendata.element_type_structure,
                     expected_element_type)
    data = synthetic_cliendata.create_tf_dataset_for_client(
        synthetic_cliendata.client_ids[0])
    self.assertLen(list(data), 3)
    image_shapes = [element['image/decoded'].shape for element in data]
    self.assertEqual(image_shapes, [synthetic_image_shape] * 3)
    labels = [element['class'] for element in data]
    expected_labels = [[1], [1], [1]]
    self.assertEqual(labels, expected_labels)

  @parameterized.named_parameters(
      ('train', lambda element: landmark._map_fn(element, is_training=True)),
      ('test', lambda element: landmark._map_fn(element, is_training=False)))
  def test_map_fn_returns_desired_image_shape_and_dtype(self, map_fn):
    synthetic_data = landmark._get_synthetic_landmark_data()
    mapped_data = synthetic_data.map(map_fn)
    image, label = next(iter(mapped_data))
    self.assertListEqual(image.shape.as_list(),
                         [landmark._IMAGE_SIZE, landmark._IMAGE_SIZE, 3])
    self.assertListEqual(label.shape.as_list(), [1])
    self.assertEqual(image.dtype, tf.float32)
    self.assertEqual(label.dtype, tf.int64)

  def _check_same_element_different_loop(self, data):
    first_element_loop_1 = iter(data).next()
    first_element_loop_2 = iter(data).next()
    tf.nest.map_structure(self.assertAllEqual, first_element_loop_1,
                          first_element_loop_2)

  def test_create_model_and_data(self):
    train_batch_size = 1
    model_fn, datasets, train_preprocess_fn, split_data_fn, accuracy_name = (
        landmark.create_model_and_data(
            num_local_epochs=1,
            train_batch_size=train_batch_size,
            use_synthetic_data=True))
    self.assertEqual(accuracy_name, landmark._ACCURACY_NAME)
    model = model_fn()
    self.assertIsInstance(model, tff.learning.Model)
    self.assertIn(accuracy_name, model.report_local_unfinalized_metrics())
    self.assertEqual(
        list(datasets.keys()), [
            constants.TRAIN_CLIENTS_KEY, constants.VALID_CLIENTS_KEY,
            constants.TEST_CLIENTS_KEY
        ])
    train_client_data = datasets[constants.TRAIN_CLIENTS_KEY]
    valid_client_data = datasets[constants.VALID_CLIENTS_KEY]
    test_client_data = datasets[constants.TEST_CLIENTS_KEY]
    for client_data in [train_client_data, valid_client_data, test_client_data]:
      self.assertIsInstance(client_data, tff.simulation.datasets.ClientData)
    # Assert that we can train the model on a single client's data.
    train_data_for_first_client = train_preprocess_fn(
        train_client_data.create_tf_dataset_for_client(
            train_client_data.client_ids[0]))
    train_batch_for_first_client = iter(train_data_for_first_client).next()
    batch_output = model.forward_pass(train_batch_for_first_client)
    self.assertEqual(batch_output.num_examples, train_batch_size)
    # Assert that we can split the validation and test clients' data.
    for client_data in [valid_client_data, test_client_data]:
      client_data_before_split = client_data.create_tf_dataset_for_client(
          client_data.client_ids[0])
      client_data_after_split = split_data_fn(client_data_before_split)
      self.assertEqual(
          list(client_data_after_split.keys()),
          [constants.PERSONALIZATION_DATA_KEY, constants.TEST_DATA_KEY])
      personalization_data = client_data_after_split[
          constants.PERSONALIZATION_DATA_KEY]
      test_data = client_data_after_split[constants.TEST_DATA_KEY]
      # Verify that every time we loop over the data, it gives the same result.
      self._check_same_element_different_loop(personalization_data)
      self._check_same_element_different_loop(test_data)
      # Before splitting, the client's local dataset has 3 examples, after
      # splitting, the personalization dataset should have 1 example and the
      # eval set should have 2 examples.
      expected_size_before_split = 3
      expected_personalization_size = 1
      expected_test_size = 2
      self.assertLen(list(client_data_before_split), expected_size_before_split)
      self.assertLen(list(personalization_data), expected_personalization_size)
      self.assertLen(list(test_data), expected_test_size)
      # Assert that `model.forward_pass` works on both datasets after splitting.
      for data in [personalization_data, test_data]:
        first_batch = iter(data.batch(train_batch_size)).next()
        batch_output = model.forward_pass(first_batch)
        self.assertEqual(batch_output.num_examples, train_batch_size)


if __name__ == '__main__':
  tf.test.main()
